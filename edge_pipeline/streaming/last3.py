import argparse
import csv
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
import queue
import signal
import json
import serial
import pynmea2

import cv2
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

# Capture interval: 3 seconds
CAPTURE_INTERVAL_S = 3.0

# GPS Config
GPS_PORT = "/dev/ttyAMA0"
GPS_BAUD = 9600
MIN_SATS = 4
MAX_HDOP = 3.0

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

# ==================================================================
# GPS Kalman Filter Classes
# ==================================================================
class Kalman1D:
    def __init__(self, Q=1e-5, R=0.005):
        self.Q = Q
        self.R = R
        self.P = 1.0
        self.x = None

    def update(self, z, hdop=1.0):
        if self.x is None:
            self.x = z
            return self.x
        self.P += self.Q
        R_scaled = self.R * (hdop ** 2)
        K = self.P / (self.P + R_scaled)
        self.x += K * (z - self.x)
        self.P *= (1 - K)
        return self.x

class GPSReader:
    """GPS reader with Kalman filtering for GY-GPS6MV2 NEO-7M."""
    def __init__(self, port=GPS_PORT, baud=GPS_BAUD):
        self.port = port
        self.baud = baud
        self.ser = None
        self.running = True
        self.lock = threading.Lock()
        
        # Kalman filters
        self.lat_k = Kalman1D()
        self.lon_k = Kalman1D()
        self.alt_k = Kalman1D(Q=1e-4, R=0.1)
        
        # Current GPS data
        self.current_data = {
            "lat": None,
            "lon": None,
            "alt": None,
            "lat_raw": None,
            "lon_raw": None,
            "sats": 0,
            "hdop": 0,
            "fix_type": 0,
            "valid": False,
            "timestamp": None,
            "fix_count": 0
        }
        
        # Statistics
        self.fix_count = 0
        self.skip_count = 0
        
        # Start GPS thread
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()
        print("GPS Reader started with Kalman filtering.")
    
    def _run(self):
        """Main GPS reading loop with Kalman filtering."""
        try:
            self.ser = serial.Serial(self.port, self.baud, timeout=1)
            print(f"GPS connected on {self.port}")
        except Exception as e:
            print(f"ERROR: Could not open GPS port {self.port}: {e}")
            return
        
        while self.running:
            try:
                raw = self.ser.readline()
                if not raw:
                    continue
                    
                line = raw.decode("ascii", errors="ignore").strip()
                
                # Look for GGA sentences
                if "GGA" in line:
                    try:
                        msg = pynmea2.parse(line)
                        if isinstance(msg, pynmea2.GGA):
                            fix = int(msg.gps_qual or 0)
                            sats = int(msg.num_sats or 0)
                            hdop = float(msg.horizontal_dil or 99)
                            lat = msg.latitude
                            lon = msg.longitude
                            alt = float(msg.altitude or 0)
                            
                            # Validate fix
                            if fix == 0 or lat == 0.0:
                                self.skip_count += 1
                                continue
                            
                            if sats < MIN_SATS:
                                self.skip_count += 1
                                continue
                            
                            # NOTE: HDOP is NOT used as a hard reject filter.
                            # High HDOP readings are still accepted but Kalman
                            # R is scaled by hdop² so noisy fixes have less weight.
                            # This matches gps2.py behaviour and prevents NO FIX
                            # when hdop briefly exceeds MAX_HDOP.
                            
                            # Update Kalman filters
                            lat_f = self.lat_k.update(lat, hdop)
                            lon_f = self.lon_k.update(lon, hdop)
                            alt_f = self.alt_k.update(alt, hdop)
                            self.fix_count += 1
                            
                            with self.lock:
                                self.current_data = {
                                    "lat": round(lat_f, 7),
                                    "lon": round(lon_f, 7),
                                    "alt": round(alt_f, 2),
                                    "lat_raw": round(lat, 7),
                                    "lon_raw": round(lon, 7),
                                    "sats": sats,
                                    "hdop": hdop,
                                    "fix_type": fix,
                                    "valid": True,
                                    "timestamp": datetime.now(timezone.utc).isoformat(),
                                    "fix_count": self.fix_count
                                }
                            
                            # Print occasional status
                            if self.fix_count % 5 == 0:
                                print(f"GPS Fix #{self.fix_count}: {lat_f:.6f}, {lon_f:.6f} | sats={sats}, hdop={hdop:.1f}")
                            
                    except (pynmea2.ParseError, ValueError):
                        pass  # malformed NMEA sentence, skip
                        
            except Exception as e:
                # Serial error — wait briefly then retry
                time.sleep(0.5)
        
        if self.ser:
            self.ser.close()
    
    def get_gps(self):
        """Get the current GPS data (thread-safe)."""
        with self.lock:
            return self.current_data.copy()
    
    def stop(self):
        """Stop the GPS thread."""
        self.running = False
        if self.thread.is_alive():
            self.thread.join(timeout=2.0)
        if self.ser:
            self.ser.close()
        print(f"GPS Reader stopped. Fixes: {self.fix_count}, Skipped: {self.skip_count}")

# ==================================================================

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

def save_frame_data(save_dir, frame_idx, rgb_frame, thermal_frame, 
                    panel_results, rgb_label, rgb_conf,
                    thermal_label, thermal_conf, gps_data):
    """Save frame data with all information."""
    try:
        # Create directory for this frame
        frame_dir = save_dir / f"frame_{frame_idx:06d}"
        frame_dir.mkdir(parents=True, exist_ok=True)
        
        # Save raw images
        cv2.imwrite(str(frame_dir / "rgb.jpg"), rgb_frame)
        cv2.imwrite(str(frame_dir / "thermal.jpg"), thermal_frame)
        
        # Create annotated image
        annotated = rgb_frame.copy()
        
        # Draw panel boxes and labels
        for panel in panel_results:
            if panel["box"]:
                x1, y1, x2, y2 = panel["box"]
                color = (0, 0, 255) if panel["label"] == "Anomaly" else (0, 255, 0)
                cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
                label = f"P{panel['panel_id']}: {panel['label']} ({panel['anomaly_prob']:.2f})"
                cv2.putText(annotated, label, (x1, y1-10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        # Add thermal label
        cv2.putText(annotated, f"Thermal: {thermal_label} ({thermal_conf:.3f})", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Add RGB label
        cv2.putText(annotated, f"RGB: {rgb_label} ({rgb_conf:.3f})", (10, 60),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Add GPS info if available
        if gps_data and gps_data["valid"]:
            gps_text = f"GPS: {gps_data['lat']:.6f}, {gps_data['lon']:.6f} | sats={gps_data['sats']}, hdop={gps_data['hdop']:.1f}"
        else:
            gps_text = "GPS: No fix"
        cv2.putText(annotated, gps_text, (10, 90),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        cv2.imwrite(str(frame_dir / "annotated.jpg"), annotated)
        
        # Save metadata with all information
        with open(frame_dir / "metadata.txt", "w") as f:
            f.write(f"Frame: {frame_idx}\n")
            f.write(f"Timestamp: {datetime.now().isoformat()}\n")
            f.write("\n=== DETECTION RESULTS ===\n")
            f.write(f"Number of Panels: {len(panel_results)}\n")
            f.write(f"RGB Label: {rgb_label}\n")
            f.write(f"RGB Confidence: {rgb_conf:.4f}\n")
            f.write(f"Thermal Label: {thermal_label}\n")
            f.write(f"Thermal Confidence: {thermal_conf:.4f}\n")
            
            f.write("\n=== PANEL DETAILS ===\n")
            for panel in panel_results:
                f.write(f"  Panel {panel['panel_id']}: {panel['label']} (prob: {panel['anomaly_prob']:.4f}, conf: {panel['confidence']:.4f})\n")
            
            f.write("\n=== GPS DATA ===\n")
            if gps_data and gps_data["valid"]:
                f.write(f"Latitude (filtered): {gps_data['lat']:.7f}\n")
                f.write(f"Longitude (filtered): {gps_data['lon']:.7f}\n")
                f.write(f"Altitude (filtered): {gps_data['alt']:.2f}m\n")
                f.write(f"Raw Latitude: {gps_data['lat_raw']:.7f}\n")
                f.write(f"Raw Longitude: {gps_data['lon_raw']:.7f}\n")
                f.write(f"Satellites: {gps_data['sats']}\n")
                f.write(f"HDOP: {gps_data['hdop']:.2f}\n")
                f.write(f"Fix Type: {gps_data['fix_type']} ({'DGPS' if gps_data['fix_type']==2 else 'GPS'})\n")
                f.write(f"GPS Timestamp: {gps_data['timestamp']}\n")
                f.write(f"Fix Count: {gps_data['fix_count']}\n")
            else:
                f.write("No valid GPS fix available\n")
        
        # Save GPS as JSON
        if gps_data and gps_data["valid"]:
            with open(frame_dir / "gps.json", "w") as f:
                json.dump(gps_data, f, indent=2)
        
        return True
    except Exception as e:
        print(f"Warning: Failed to save frame {frame_idx}: {e}")
        return False

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--duration", type=int, default=None,
                         help="Test duration in seconds (default: run until Ctrl+C)")
    parser.add_argument("--disable-gps", action="store_true",
                         help="Disable GPS")
    args = parser.parse_args()

    # Create run folder with date and time
    run_folder_name = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_save_dir = Path("/home/pi5/Documents/anomalies")
    save_dir = base_save_dir / run_folder_name
    save_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Run folder: {save_dir}")
    print(f"Capture interval: {CAPTURE_INTERVAL_S}s")
    print(f"GPS: {'DISABLED' if args.disable_gps else 'ENABLED'}")
    if not args.disable_gps:
        # Updated banner to reflect actual behavior - HDOP is used for Kalman weighting, not as a hard cap
        print(f"GPS Criteria: MIN_SATS={MIN_SATS}, HDOP-weighted Kalman (no hard cap)")

    # Start GPS (unless disabled)
    gps_reader = None
    if not args.disable_gps:
        try:
            gps_reader = GPSReader()
            time.sleep(2)  # Give GPS time to get first fix
        except Exception as e:
            print(f"Warning: GPS initialization failed: {e}")
            print("Continuing without GPS...")
            gps_reader = None
    else:
        print("GPS disabled by user.")

    # Create queues
    rgb_queue = queue.Queue(maxsize=4)
    thermal_queue = queue.Queue(maxsize=4)

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

    # CSV log
    csv_path = save_dir / "detection_log.csv"
    
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "frame", "n_panels", "rgb_label", "rgb_conf", 
            "thermal_label", "thermal_conf", 
            "gps_lat", "gps_lon", "gps_alt", "gps_sats", "gps_hdop", "gps_valid", "saved"
        ])

        start_time = time.time()
        frame_idx = 0
        saved_count = 0
        
        print(f"\nStarting capture (interval={CAPTURE_INTERVAL_S}s)...")
        print("   Press Ctrl+C to stop.")
        print("\n" + "="*120)
        print(f"{'Frame':>6} {'Panels':>6} {'RGB Label':>12} {'Thermal':>12} {'GPS Status':>25} {'Saved':>6}")
        print("="*120)
        
        try:
            while running:
                loop_t0 = time.time()
                
                # Get GPS data
                gps_data = None
                if gps_reader:
                    gps_data = gps_reader.get_gps()
                
                # Get frames
                try:
                    rgb_ts, rgb_frame = rgb_queue.get(timeout=0.5)
                    thermal_ts, thermal_frame = thermal_queue.get(timeout=0.5)
                except queue.Empty:
                    continue
                
                # Process RGB
                rgb_result = rgb_pipeline.process(rgb_frame)
                n_panels = rgb_result["n_panels"]
                
                # Get RGB label (overall classification)
                if n_panels > 0:
                    # Get the worst panel label (anomaly > uncertain > healthy)
                    labels = [p["label"] for p in rgb_result["panel_results"]]
                    if "Anomaly" in labels:
                        rgb_label = "Anomaly"
                        rgb_conf = max([p["anomaly_prob"] for p in rgb_result["panel_results"] if p["label"] == "Anomaly"], default=0.0)
                    elif "Uncertain" in labels:
                        rgb_label = "Uncertain"
                        rgb_conf = max([p["anomaly_prob"] for p in rgb_result["panel_results"] if p["label"] == "Uncertain"], default=0.0)
                    else:
                        rgb_label = "Healthy"
                        rgb_conf = max([p["confidence"] for p in rgb_result["panel_results"]], default=0.0)
                else:
                    rgb_label = "No Panels"
                    rgb_conf = 0.0
                
                # Run thermal classification if panels detected
                thermal_label = "SKIPPED"
                thermal_conf = 0.0
                
                if n_panels > 0:
                    thermal_label, thermal_conf, _, _ = thermal_pipeline.predict(thermal_frame)
                
                # Save every frame
                saved = save_frame_data(
                    save_dir, frame_idx, rgb_frame, thermal_frame,
                    rgb_result["panel_results"], rgb_label, rgb_conf,
                    thermal_label, thermal_conf, gps_data
                )
                if saved:
                    saved_count += 1
                
                # GPS status for display
                if gps_data and gps_data["valid"]:
                    gps_status = f"{gps_data['sats']:2d}sat hdop={gps_data['hdop']:.1f} {gps_data['lat']:.5f},{gps_data['lon']:.5f}"
                else:
                    gps_status = "NO FIX (waiting...)"
                
                # Display
                thermal_display = thermal_label[:12]  # Truncate for display
                rgb_display = rgb_label[:12]  # Truncate for display
                saved_indicator = "YES" if saved else "NO"
                
                print(f"{frame_idx:6d} {n_panels:6d} {rgb_display:>12} {thermal_display:>12} {gps_status:>25} {saved_indicator:>6}")
                
                # Log to CSV with GPS data
                writer.writerow([
                    frame_idx,
                    n_panels,
                    rgb_label,
                    f"{rgb_conf:.4f}",
                    thermal_label,
                    f"{thermal_conf:.4f}",
                    f"{gps_data['lat']:.7f}" if gps_data and gps_data["valid"] else "",
                    f"{gps_data['lon']:.7f}" if gps_data and gps_data["valid"] else "",
                    f"{gps_data['alt']:.2f}" if gps_data and gps_data["valid"] else "",
                    gps_data["sats"] if gps_data and gps_data["valid"] else 0,
                    f"{gps_data['hdop']:.2f}" if gps_data and gps_data["valid"] else "",
                    gps_data["valid"] if gps_data else False,
                    saved
                ])
                f.flush()
                
                frame_idx += 1
                
                # Check duration if specified
                if args.duration and (time.time() - start_time) >= args.duration:
                    print(f"\nDuration {args.duration}s completed.")
                    break
                
                # Wait for next capture (3 seconds)
                elapsed_loop = time.time() - loop_t0
                sleep_left = CAPTURE_INTERVAL_S - elapsed_loop
                if sleep_left > 0:
                    time.sleep(sleep_left)
                
        except KeyboardInterrupt:
            print("\nInterrupted by user.")
        finally:
            # Clean up
            running = False
            rgb_capture.stop()
            thermal_capture.stop()
            if gps_reader:
                gps_reader.stop()
            
            print("\n" + "="*120)
            print(f"Capture completed.")
            print(f"   Run folder: {save_dir}")
            print(f"   Total frames processed: {frame_idx}")
            print(f"   Frames saved: {saved_count}")
            if gps_reader:
                print(f"   GPS fixes: {gps_reader.fix_count}")
                print(f"   GPS skipped: {gps_reader.skip_count}")
            print(f"   CSV log: {csv_path}")
            print("="*120)

if __name__ == "__main__":
    main()
