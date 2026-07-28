import argparse
import csv
import os
import subprocess
import threading
import time
from datetime import datetime
from pathlib import Path
import queue
import signal

import cv2
import psutil
import numpy as np

# ====================== RGB IMPORTS ======================
import onnxruntime as ort
from ultralytics import YOLO

# ====================== THERMAL IMPORTS ======================
try:
    from ai_edge_litert.interpreter import Interpreter
    print("Using ai_edge_litert for TFLite")
except ImportError:
    try:
        from tflite_runtime.interpreter import Interpreter
        print("Using tflite_runtime for TFLite")
    except ImportError:
        try:
            from tensorflow.lite.python.interpreter import Interpreter
            print("Using tensorflow.lite for TFLite")
        except ImportError:
            try:
                import tensorflow as tf
                Interpreter = tf.lite.Interpreter
                print("Using tensorflow.lite.Interpreter")
            except ImportError:
                raise ImportError(
                    "No TFLite interpreter found. Please install one of:\n"
                    "pip install ai-edge-litert\n"
                    "pip install tflite-runtime\n"
                    "pip install tensorflow"
                )

# ==================================================================
# CONFIG
# ==================================================================
RGB_DEVICE = "/dev/video2"
THERMAL_DEVICE = "/dev/video0"
THERMAL_W, THERMAL_H = 256, 192

YOLO_MODEL_PATH = "/home/pi5/Documents/panel_detection2/best_ncnn_model"
CLASSIFIER_ONNX_PATH = "/home/pi5/Documents/binary_class/onnx_128/classifier_opt.onnx"
THERMAL_BINARY_MODEL_PATH = "/home/pi5/Documents/thermal_binary/thermal_binary_mobilenetv2_f16.tflite"

YOLO_CONF = 0.35
YOLO_IOU = 0.45
YOLO_IMGSZ = 640
CLASSIFIER_IMGSZ = 128

ANOMALY_THRESHOLD = 0.60
UNCERTAIN_BAND = 0.10
THERMAL_THRESHOLD = 0.5

FLIGHT_TRIGGER_INTERVAL_S = 1.875

_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(1, 3, 1, 1)
_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(1, 3, 1, 1)
_CLAHE_OBJ = cv2.createCLAHE(clipLimit=1.5, tileGridSize=(4, 4))

# ==================================================================

def apply_clahe_fast(img):
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    lab[:,:,0] = _CLAHE_OBJ.apply(lab[:,:,0])
    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

def preprocess_batch(crops, size=128):
    if not crops:
        return np.empty((0, 3, size, size), dtype=np.float32)
    n = len(crops)
    batch = np.empty((n, 3, size, size), dtype=np.float32)
    for i, crop in enumerate(crops):
        resized = cv2.resize(crop, (size, size))
        batch[i] = np.ascontiguousarray(resized[:, :, ::-1]).transpose(2, 0, 1)
    batch = batch / 255.0
    batch -= _MEAN
    batch /= _STD
    return batch

def softmax(x):
    e = np.exp(x - x.max())
    return e / e.sum()

class RGBPipeline:
    def __init__(self):
        print("Loading RGB models...")
        self.yolo = YOLO(YOLO_MODEL_PATH, task="obb")
        sess_options = ort.SessionOptions()
        sess_options.intra_op_num_threads = 2
        sess_options.inter_op_num_threads = 1
        self.classifier = ort.InferenceSession(
            CLASSIFIER_ONNX_PATH, 
            sess_options=sess_options,
            providers=["CPUExecutionProvider"]
        )
        self.cls_input_name = self.classifier.get_inputs()[0].name
        print("RGB models loaded.")

    def process(self, rgb_frame):
        t0 = time.perf_counter()
        clahe_img = apply_clahe_fast(rgb_frame)
        
        results = self.yolo.predict(
            clahe_img, 
            imgsz=YOLO_IMGSZ, 
            conf=YOLO_CONF, 
            iou=YOLO_IOU,
            half=False, 
            augment=False, 
            verbose=False
        )
        
        crops = []
        h, w = clahe_img.shape[:2]
        panel_results = []
        boxes = []

        if results and results[0].obb is not None:
            for r in results:
                for box in r.obb.xyxyxyxy.cpu().numpy():
                    pts = box.reshape(4, 2)
                    x1 = max(0, int(pts[:,0].min()))
                    y1 = max(0, int(pts[:,1].min()))
                    x2 = min(w, int(pts[:,0].max()))
                    y2 = min(h, int(pts[:,1].max()))
                    if x2 > x1 and y2 > y1:
                        crops.append(clahe_img[y1:y2, x1:x2])
                        boxes.append((x1, y1, x2, y2))

            if crops:
                batch = preprocess_batch(crops, CLASSIFIER_IMGSZ)
                logits = self.classifier.run(None, {self.cls_input_name: batch})[0]
                for i, logit in enumerate(logits):
                    probs = softmax(logit)
                    anomaly_prob = float(probs[1])
                    label = "Anomaly" if anomaly_prob >= ANOMALY_THRESHOLD else \
                            "Uncertain" if anomaly_prob >= (ANOMALY_THRESHOLD - UNCERTAIN_BAND) else "Healthy"
                    panel_results.append({
                        "panel_id": i+1,
                        "label": label,
                        "anomaly_prob": anomaly_prob,
                        "confidence": float(probs.max()),
                        "box": boxes[i] if i < len(boxes) else None
                    })

        total_ms = (time.perf_counter() - t0) * 1000
        return {
            "panel_results": panel_results,
            "total_ms": total_ms,
            "n_panels": len(panel_results)
        }

class ThermalPipeline:
    def __init__(self, model_path=THERMAL_BINARY_MODEL_PATH, threshold=THERMAL_THRESHOLD):
        print(f"Loading thermal model from {model_path}...")
        self.interpreter = Interpreter(model_path=str(model_path))
        self.interpreter.allocate_tensors()
        self.threshold = threshold
        self.input_details = self.interpreter.get_input_details()[0]
        self.output_details = self.interpreter.get_output_details()[0]
        self.input_size = 224
        print("Thermal model loaded.")

    def preprocess(self, image):
        if len(image.shape) == 3 and image.shape[2] == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        resized = cv2.resize(image, (self.input_size, self.input_size))
        normalized = (resized.astype(np.float32) / 127.5) - 1.0
        return np.expand_dims(normalized, axis=0)

    def predict(self, thermal_frame):
        t0 = time.perf_counter()
        input_tensor = self.preprocess(thermal_frame)
        self.interpreter.set_tensor(self.input_details['index'], input_tensor)
        self.interpreter.invoke()
        prob = float(self.interpreter.get_tensor(self.output_details['index'])[0][0])
        elapsed_ms = (time.perf_counter() - t0) * 1000
        
        if prob < self.threshold:
            return "ANOMALY", 1 - prob, prob, elapsed_ms
        return "NO_ANOMALY", prob, prob, elapsed_ms

class CaptureThread(threading.Thread):
    def __init__(self, device, queue, format_yuyv=False, width=640, height=480):
        super().__init__(daemon=True)
        self.device = device
        self.queue = queue
        self.format_yuyv = format_yuyv
        self.width = width
        self.height = height
        self.running = True
        self.cap = None

    def run(self):
        self.cap = cv2.VideoCapture(self.device, cv2.CAP_V4L2)
        if not self.cap.isOpened():
            print(f"Error: Could not open {self.device}")
            return
        
        if self.format_yuyv:
            self.cap.set(cv2.CAP_PROP_CONVERT_RGB, 0)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        
        while self.running:
            ret, frame = self.cap.read()
            if ret:
                if self.format_yuyv:
                    frame = cv2.cvtColor(frame, cv2.COLOR_YUV2BGR_YUYV)
                    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    frame = cv2.applyColorMap(gray, cv2.COLORMAP_INFERNO)
                try:
                    self.queue.put((time.time(), frame), block=False)
                except queue.Full:
                    pass
            time.sleep(0.001)
        
        self.cap.release()

    def stop(self):
        self.running = False
        if self.cap:
            self.cap.release()

def create_simple_display(rgb_frame, thermal_frame):
    """Create a simple side-by-side display of raw camera feeds."""
    h_rgb = rgb_frame.shape[0]
    thermal_resized = cv2.resize(thermal_frame, 
                                 (int(thermal_frame.shape[1] * h_rgb / thermal_frame.shape[0]), 
                                  h_rgb))
    combined = np.hstack([rgb_frame, thermal_resized])
    h, w = combined.shape[:2]
    cv2.putText(combined, "RGB Camera", (10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(combined, "Thermal Camera", (rgb_frame.shape[1] + 10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    return combined

def save_frame_pair(pair_dir, frame_idx, rgb_frame, thermal_frame, elapsed, panel_results, thermal_label, thermal_skipped):
    """Save the current frame pair to disk."""
    try:
        save_dir = pair_dir / f"frame_{frame_idx:06d}"
        save_dir.mkdir(parents=True, exist_ok=True)
        
        cv2.imwrite(str(save_dir / "rgb_raw.jpg"), rgb_frame)
        cv2.imwrite(str(save_dir / "thermal.jpg"), thermal_frame)
        
        combined = create_simple_display(rgb_frame, thermal_frame)
        
        # Add classification results to the combined image
        if thermal_skipped:
            info_text = f"Frame: {frame_idx} | Thermal: SKIPPED (no panels) | Panels: {len(panel_results)}"
        else:
            info_text = f"Frame: {frame_idx} | Thermal: {thermal_label} | Panels: {len(panel_results)}"
        cv2.putText(combined, info_text, (10, combined.shape[0] - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Add per-panel results
        y_offset = 60
        for i, panel in enumerate(panel_results[:5]):  # Show first 5 panels
            if i < 5:
                panel_text = f"P{panel['panel_id']}: {panel['label']} ({panel['anomaly_prob']:.2f})"
                cv2.putText(combined, panel_text, (10, y_offset + i*25),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        cv2.imwrite(str(save_dir / "combined_with_results.jpg"), combined)
        
        with open(save_dir / "metadata.txt", "w") as f:
            f.write(f"Frame: {frame_idx}\n")
            f.write(f"Timestamp: {datetime.now().isoformat()}\n")
            f.write(f"Elapsed: {elapsed:.2f}s\n")
            if thermal_skipped:
                f.write(f"Thermal: SKIPPED (no panels detected)\n")
            else:
                f.write(f"Thermal Label: {thermal_label}\n")
            f.write(f"Number of Panels: {len(panel_results)}\n")
            f.write("\nPanel Results:\n")
            for panel in panel_results:
                f.write(f"  Panel {panel['panel_id']}: {panel['label']} (prob: {panel['anomaly_prob']:.3f})\n")
        
        return True
    except Exception as e:
        print(f"Warning: Failed to save frame {frame_idx}: {e}")
        return False

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--duration", type=int, default=1200,
                         help="Test duration in seconds (default 1200 = 20 min)")
    parser.add_argument("--mode", choices=["flight", "maxrate"], default="flight",
                         help="flight = pace captures at the real trigger interval; "
                              "maxrate = capture as fast as possible (worst-case load)")
    parser.add_argument("--output", type=str, default=None,
                         help="CSV output file for results (optional)")
    parser.add_argument("--save-frames", action="store_true",
                         help="Save frame pairs to disk for visualization")
    parser.add_argument("--save-anomalies-only", action="store_true",
                         help="Only save frames with anomalies (requires --save-frames)")
    args = parser.parse_args()

    # Create log directory
    run_dir = Path("logs") / datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir.mkdir(parents=True, exist_ok=True)
    
    # CSV output
    if args.output:
        csv_path = Path(args.output)
    else:
        csv_path = run_dir / "results.csv"
    
    # Create directory for saved frames
    if args.save_frames:
        pairs_dir = run_dir / "saved_pairs"
        pairs_dir.mkdir(parents=True, exist_ok=True)
        print(f"Saving frame pairs to: {pairs_dir}/")
    
    print(f"Results will be saved to: {csv_path}")
    print(f"Mode: {args.mode}, Duration: {args.duration}s")
    print(f"Save Frames: {'ON' if args.save_frames else 'OFF'}")
    if args.save_frames and args.save_anomalies_only:
        print("   (Saving anomalies only)")

    # Create queues
    rgb_queue = queue.Queue(maxsize=8)
    thermal_queue = queue.Queue(maxsize=8)

    # Start capture threads
    rgb_capture = CaptureThread(RGB_DEVICE, rgb_queue, format_yuyv=False, width=640, height=480)
    thermal_capture = CaptureThread(THERMAL_DEVICE, thermal_queue, format_yuyv=True, width=256, height=192)
    rgb_capture.start()
    thermal_capture.start()
    print("Capture threads started.")

    # Load models
    rgb_pipeline = RGBPipeline()
    thermal_pipeline = ThermalPipeline(THERMAL_BINARY_MODEL_PATH, threshold=THERMAL_THRESHOLD)

    # Signal handler
    running = True
    def signal_handler(sig, frame):
        nonlocal running
        running = False
        print("\nShutting down...")
    signal.signal(signal.SIGINT, signal_handler)

    # Open CSV
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "elapsed_s", "frame_idx", "rgb_processing_ms", 
            "n_panels", "n_anomalies", "thermal_label", 
            "thermal_conf", "thermal_ms", "loop_total_ms", "fps",
            "thermal_skipped"
        ])

        start_time = time.time()
        frame_idx = 0
        frame_pairs_processed = 0
        saved_count = 0
        thermal_skipped_count = 0
        
        print(f"\nStarting stress test (mode={args.mode})...")
        print(f"   Trigger interval: {FLIGHT_TRIGGER_INTERVAL_S}s")
        print("   Thermal classification runs ONLY when panels are detected")
        print("   Press Ctrl+C to stop.")
        print("\n" + "="*95)
        print(f"{'Frame':>6} {'Elapsed':>8} {'RGB(ms)':>8} {'Panels':>6} {'Anom':>5} {'Thermal':>10} {'Conf':>6} {'FPS':>6} {'Saved':>6} {'ThrmSkip':>8}")
        print("="*95)
        
        try:
            while running and (time.time() - start_time) < args.duration:
                loop_t0 = time.time()
                elapsed = time.time() - start_time
                
                # Get paired frames
                try:
                    rgb_ts, rgb_frame = rgb_queue.get(timeout=0.1)
                    thermal_ts, thermal_frame = thermal_queue.get(timeout=0.1)
                except queue.Empty:
                    continue
                
                # Process RGB
                rgb_result = rgb_pipeline.process(rgb_frame)
                n_panels = rgb_result["n_panels"]
                
                # Only run thermal classification if panels are detected
                thermal_skipped = False
                th_label = "SKIPPED"
                th_conf = 0.0
                th_ms = 0.0
                
                if n_panels > 0:
                    th_label, th_conf, th_anomaly_prob, th_ms = thermal_pipeline.predict(thermal_frame)
                else:
                    thermal_skipped = True
                    thermal_skipped_count += 1
                
                # Count anomalies (only from RGB panels, thermal is separate)
                n_anomalies = sum(1 for p in rgb_result["panel_results"] if p["label"] == "Anomaly")
                has_anomaly = (n_anomalies > 0 or (th_label == "ANOMALY" and not thermal_skipped))
                
                # Calculate timings
                loop_total_ms = (time.time() - loop_t0) * 1000
                fps = 1000.0 / loop_total_ms if loop_total_ms > 0 else float("inf")
                
                # Save frame pairs if enabled
                saved_this_frame = False
                if args.save_frames:
                    should_save = False
                    if args.save_anomalies_only and has_anomaly:
                        should_save = True
                    elif not args.save_anomalies_only:
                        should_save = True
                    
                    if should_save:
                        if save_frame_pair(pairs_dir, frame_idx, rgb_frame, thermal_frame, 
                                         elapsed, rgb_result["panel_results"], th_label, thermal_skipped):
                            saved_count += 1
                            saved_this_frame = True
                
                # Output to terminal
                saved_indicator = "YES" if saved_this_frame else "NO"
                thermal_skip_indicator = "YES" if thermal_skipped else "NO"
                
                # Format thermal label for display
                if thermal_skipped:
                    thermal_display = " SKIPPED"
                else:
                    thermal_display = f"{th_label:>10}"
                
                print(f"{frame_idx:6d} {elapsed:8.2f} {rgb_result['total_ms']:8.2f} "
                      f"{n_panels:6d} {n_anomalies:5d} "
                      f"{thermal_display} {th_conf:6.3f} {fps:6.1f} {saved_indicator:>6} {thermal_skip_indicator:>8}")
                
                # Write to CSV
                writer.writerow([
                    f"{elapsed:.2f}", frame_idx,
                    f"{rgb_result['total_ms']:.2f}",
                    n_panels,
                    n_anomalies,
                    th_label,
                    f"{th_conf:.4f}",
                    f"{th_ms:.2f}",
                    f"{loop_total_ms:.2f}",
                    f"{fps:.2f}",
                    thermal_skipped
                ])
                f.flush()
                
                frame_idx += 1
                frame_pairs_processed += 1
                
                # Flight mode pacing
                if args.mode == "flight":
                    elapsed_loop = time.time() - loop_t0
                    sleep_left = FLIGHT_TRIGGER_INTERVAL_S - elapsed_loop
                    if sleep_left > 0:
                        time.sleep(sleep_left)
                
        except KeyboardInterrupt:
            print("\nInterrupted by user.")
        finally:
            # Clean up
            running = False
            rgb_capture.stop()
            thermal_capture.stop()
            
            print("\n" + "="*95)
            print(f"Test completed.")
            print(f"   Processed {frame_pairs_processed} frame pairs")
            print(f"   Thermal skipped: {thermal_skipped_count} frames (no panels detected)")
            print(f"   Thermal executed: {frame_pairs_processed - thermal_skipped_count} frames")
            print(f"   Saved {saved_count} frame pairs to disk")
            print(f"   Results saved to: {csv_path}")
            if args.save_frames:
                print(f"   Saved frames: {pairs_dir}/")
            print("="*95)

if __name__ == "__main__":
    main()
