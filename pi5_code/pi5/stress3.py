#!/usr/bin/env python3
"""
Pi5 onboard pipeline stress test with HTTP streaming.

Runs continuous (or flight-paced) dual-camera capture + onboard inference on
the Raspberry Pi 5 for a sustained duration, while independently logging
thermal/CPU/memory metrics. Streams video feed over HTTP so you can view
it from your Mac browser via SSH tunnel.

Usage:
    python3 stress_test.py --duration 1200 --mode flight
    python3 stress_test.py --duration 1200 --mode maxrate

--mode flight   paces captures at your real flight trigger interval (1.875s)
--mode maxrate  captures as fast as possible -- a worse-case thermal/load test

Two CSV logs land in ./logs/<timestamp>/:
    system_metrics.csv   -- sampled every SYSTEM_SAMPLE_INTERVAL seconds,
                             independent of the inference loop
    pipeline_metrics.csv -- one row per processed frame pair

To view stream from Mac:
1. Run this script on Pi
2. On Mac, open terminal and create SSH tunnel:
   ssh -L 8080:localhost:8080 pi5@<pi-ip-address>
3. Open browser on Mac and go to: http://localhost:8080
"""

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
import io
import socketserver
from http.server import HTTPServer, BaseHTTPRequestHandler

import cv2
import psutil
import numpy as np

# ====================== RGB IMPORTS ======================
import onnxruntime as ort
from ultralytics import YOLO

# ====================== THERMAL IMPORTS ======================
# Try different import options for TFLite, but prefer the newer litert
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
            # Fallback: try to use the older flatbuffers-based import
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
# CONFIG -- EDIT THESE
# ==================================================================
RGB_DEVICE = "/dev/video2"          # confirm with `v4l2-ctl --list-devices`
THERMAL_DEVICE = "/dev/video0"      # TC002C, YUYV, 256x192
THERMAL_W, THERMAL_H = 256, 192

# RGB Pipeline paths
YOLO_MODEL_PATH = "/home/pi5/Documents/panel_detection2/best_ncnn_model"
CLASSIFIER_ONNX_PATH = "/home/pi5/Documents/binary_class/onnx_128/classifier_opt.onnx"

# Thermal model path
THERMAL_BINARY_MODEL_PATH = "/home/pi5/Documents/thermal_binary/thermal_binary_mobilenetv2_f16.tflite"

# RGB Pipeline config
YOLO_CONF = 0.35
YOLO_IOU = 0.45
YOLO_IMGSZ = 640
CLASSIFIER_IMGSZ = 128

ANOMALY_THRESHOLD = 0.60
UNCERTAIN_BAND = 0.10

# Thermal config
THERMAL_THRESHOLD = 0.5

# Flight config
FLIGHT_TRIGGER_INTERVAL_S = 1.875   # matches your 1.5m / 0.8m/s flight plan
SYSTEM_SAMPLE_INTERVAL_S = 2.0      # independent watchdog sampling rate

THROTTLE_TEMP_C = 80.0

# RGB preprocessing constants (matching the working code)
_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(1, 3, 1, 1)
_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(1, 3, 1, 1)
_CLAHE_OBJ = cv2.createCLAHE(clipLimit=1.5, tileGridSize=(4, 4))

# ====================== HTTP STREAMING CONFIG ======================
STREAM_PORT = 8080
STREAM_QUALITY = 50  # JPEG quality (1-100)
STREAM_FPS = 4  # Frames per second for streaming (max 4 FPS for 8080 port)

# ==================================================================

def get_cpu_temp_c():
    try:
        out = subprocess.check_output(["vcgencmd", "measure_temp"]).decode()
        return float(out.strip().replace("temp=", "").replace("'C", ""))
    except Exception:
        try:
            with open("/sys/class/thermal/thermal_zone0/temp") as f:
                return int(f.read().strip()) / 1000.0
        except Exception:
            return float("nan")


def get_throttle_flags():
    """Decode vcgencmd get_throttled bitmask."""
    try:
        out = subprocess.check_output(["vcgencmd", "get_throttled"]).decode().strip()
        val = int(out.split("=")[1], 16)
        return {
            "under_voltage_now": bool(val & 0x1),
            "freq_capped_now": bool(val & 0x2),
            "throttled_now": bool(val & 0x4),
            "soft_temp_limit_now": bool(val & 0x8),
            "throttled_since_boot": bool(val & 0x40000),
            "soft_temp_limit_since_boot": bool(val & 0x80000),
        }
    except Exception:
        return {}


def get_cpu_freq_mhz():
    try:
        out = subprocess.check_output(["vcgencmd", "measure_clock", "arm"]).decode()
        return int(out.strip().split("=")[1]) / 1e6
    except Exception:
        return float("nan")


# ====================== RGB PIPELINE ======================

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
    """Complete RGB pipeline: YOLO detection + classifier"""
    
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
        """Process RGB frame through YOLO + classifier."""
        t0 = time.perf_counter()
        clahe_img = apply_clahe_fast(rgb_frame)
        t_clahe = (time.perf_counter() - t0) * 1000

        t1 = time.perf_counter()
        results = self.yolo.predict(
            clahe_img, 
            imgsz=YOLO_IMGSZ, 
            conf=YOLO_CONF, 
            iou=YOLO_IOU,
            half=False, 
            augment=False, 
            verbose=False
        )
        t_yolo = (time.perf_counter() - t1) * 1000

        t2 = time.perf_counter()
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

        t_classify = (time.perf_counter() - t2) * 1000
        total_ms = (time.perf_counter() - t0) * 1000

        return {
            "clahe_img": clahe_img,
            "panel_results": panel_results,
            "timings": {
                "clahe": t_clahe, 
                "yolo": t_yolo, 
                "classify": t_classify, 
                "total": total_ms
            }
        }


# ====================== THERMAL PIPELINE ======================

class ThermalPipeline:
    """Thermal panel detector matching the working code"""
    
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
        """Run thermal inference. Returns (label, confidence, anomaly_prob)."""
        t0 = time.perf_counter()
        input_tensor = self.preprocess(thermal_frame)
        self.interpreter.set_tensor(self.input_details['index'], input_tensor)
        self.interpreter.invoke()
        prob = float(self.interpreter.get_tensor(self.output_details['index'])[0][0])
        elapsed_ms = (time.perf_counter() - t0) * 1000
        
        if prob < self.threshold:
            return "ANOMALY", 1 - prob, prob, elapsed_ms
        return "NO_ANOMALY", prob, prob, elapsed_ms


# ====================== SYSTEM MONITOR ======================

class SystemMonitor(threading.Thread):
    """Background watchdog for system metrics."""
    
    def __init__(self, log_path, interval_s, stop_event):
        super().__init__(daemon=True)
        self.log_path = log_path
        self.interval_s = interval_s
        self.stop_event = stop_event
        self.proc = psutil.Process(os.getpid())

    def run(self):
        with open(self.log_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                "elapsed_s", "cpu_temp_c", "cpu_freq_mhz",
                "throttled_now", "soft_temp_limit_now", "throttled_since_boot",
                "process_rss_mb", "system_mem_used_pct", "system_cpu_pct",
            ])
            start = time.time()
            while not self.stop_event.is_set():
                t = time.time() - start
                temp = get_cpu_temp_c()
                freq = get_cpu_freq_mhz()
                flags = get_throttle_flags()
                rss_mb = self.proc.memory_info().rss / (1024 * 1024)
                mem_pct = psutil.virtual_memory().percent
                cpu_pct = psutil.cpu_percent(interval=None)
                writer.writerow([
                    f"{t:.1f}", temp, freq,
                    flags.get("throttled_now"), flags.get("soft_temp_limit_now"),
                    flags.get("throttled_since_boot"),
                    f"{rss_mb:.1f}", mem_pct, cpu_pct,
                ])
                f.flush()
                if temp >= THROTTLE_TEMP_C:
                    print(f"[WARN] t={t:.0f}s  CPU temp {temp:.1f}C >= {THROTTLE_TEMP_C}C threshold")
                self.stop_event.wait(self.interval_s)


# ====================== CAPTURE THREADS ======================

class CaptureThread(threading.Thread):
    """Threaded camera capture with queue."""
    
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


# ====================== HTTP STREAMING SERVER ======================

class StreamingHandler(BaseHTTPRequestHandler):
    """HTTP handler for MJPEG streaming."""
    
    # Class variable to hold the latest frame
    latest_frame = None
    frame_lock = threading.Lock()
    
    def do_GET(self):
        if self.path == '/':
            self.send_response(200)
            self.send_header('Content-Type', 'text/html')
            self.end_headers()
            # Fixed: Use string and encode, or escape special chars
            html = """<html>
                <head>
                    <title>Pi5 Camera Stream</title>
                    <style>
                        body { 
                            background: #1a1a1a; 
                            color: white; 
                            font-family: Arial, sans-serif;
                            display: flex;
                            flex-direction: column;
                            align-items: center;
                            justify-content: center;
                            height: 100vh;
                            margin: 0;
                        }
                        h1 { color: #4CAF50; margin-bottom: 10px; }
                        .info { color: #888; margin-bottom: 20px; font-size: 14px; }
                        img { 
                            border: 2px solid #333;
                            border-radius: 8px;
                            max-width: 95%;
                            box-shadow: 0 4px 8px rgba(0,0,0,0.5);
                        }
                        .status {
                            margin-top: 20px;
                            color: #4CAF50;
                            font-size: 12px;
                        }
                    </style>
                </head>
                <body>
                    <h1>Pi5 Camera Stream</h1>
                    <div class="info">RGB (left) | Thermal (right) - Click image for full size</div>
                    <img src="/stream.mjpg" alt="Camera Stream" />
                    <div class="status">Live | FPS: ~4 | Click image to open in new tab</div>
                    <script>
                        const img = document.querySelector('img');
                        img.addEventListener('click', function() {
                            window.open(this.src, '_blank');
                        });
                    </script>
                </body>
                </html>"""
            self.wfile.write(html.encode())
        elif self.path == '/stream.mjpg':
            self.send_response(200)
            self.send_header('Content-Type', 'multipart/x-mixed-replace; boundary=--jpgboundary')
            self.end_headers()
            
            while True:
                with StreamingHandler.frame_lock:
                    frame = StreamingHandler.latest_frame
                
                if frame is not None:
                    try:
                        # Encode frame as JPEG
                        ret, jpeg = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, STREAM_QUALITY])
                        if ret:
                            self.wfile.write(b'--jpgboundary\r\n')
                            self.wfile.write(b'Content-Type: image/jpeg\r\n')
                            self.wfile.write(b'Content-Length: ' + str(len(jpeg)).encode() + b'\r\n\r\n')
                            self.wfile.write(jpeg.tobytes())
                            self.wfile.write(b'\r\n')
                    except (BrokenPipeError, ConnectionError):
                        break
                time.sleep(1.0 / STREAM_FPS)
        else:
            self.send_response(404)
            self.end_headers()

    def log_message(self, format, *args):
        # Suppress log messages for cleaner output
        pass


class StreamingServer:
    """MJPEG streaming server running in a separate thread."""
    
    def __init__(self, port=STREAM_PORT):
        self.port = port
        self.server = None
        self.thread = None
        self.running = False
        
    def start(self):
        """Start the HTTP server in a background thread."""
        self.running = True
        self.server = HTTPServer(('0.0.0.0', self.port), StreamingHandler)
        self.thread = threading.Thread(target=self._run_server, daemon=True)
        self.thread.start()
        print(f"HTTP Stream available at: http://localhost:{self.port}")
        print(f"   On Mac: ssh -L {self.port}:localhost:{self.port} pi5@<pi-ip>")
        print(f"   Then open: http://localhost:{self.port}")
        
    def _run_server(self):
        try:
            self.server.serve_forever()
        except Exception as e:
            print(f"Streaming server error: {e}")
            
    def stop(self):
        """Stop the HTTP server."""
        self.running = False
        if self.server:
            self.server.shutdown()
            self.server.server_close()
            
    def update_frame(self, frame):
        """Update the latest frame for streaming."""
        with StreamingHandler.frame_lock:
            StreamingHandler.latest_frame = frame


# ====================== VISUALIZATION HELPER ======================

def create_simple_display(rgb_frame, thermal_frame):
    """Create a simple side-by-side display of raw camera feeds."""
    # Resize thermal to match RGB height
    h_rgb = rgb_frame.shape[0]
    thermal_resized = cv2.resize(thermal_frame, 
                                 (int(thermal_frame.shape[1] * h_rgb / thermal_frame.shape[0]), 
                                  h_rgb))
    
    # Stack side by side
    combined = np.hstack([rgb_frame, thermal_resized])
    
    # Add labels
    h, w = combined.shape[:2]
    cv2.putText(combined, "RGB Camera", (10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(combined, "Thermal Camera", (rgb_frame.shape[1] + 10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    return combined


# ====================== SAVING HELPER ======================

def save_frame_pair(pair_dir, frame_idx, rgb_frame, thermal_frame, elapsed):
    """Save the current frame pair to disk."""
    try:
        save_dir = pair_dir / f"frame_{frame_idx:06d}"
        save_dir.mkdir(parents=True, exist_ok=True)
        
        cv2.imwrite(str(save_dir / "rgb_raw.jpg"), rgb_frame)
        cv2.imwrite(str(save_dir / "thermal.jpg"), thermal_frame)
        
        combined = create_simple_display(rgb_frame, thermal_frame)
        cv2.imwrite(str(save_dir / "combined.jpg"), combined)
        
        with open(save_dir / "metadata.txt", "w") as f:
            f.write(f"Frame: {frame_idx}\n")
            f.write(f"Timestamp: {datetime.now().isoformat()}\n")
            f.write(f"Elapsed: {elapsed:.2f}s\n")
            f.write(f"RGB resolution: {rgb_frame.shape[1]}x{rgb_frame.shape[0]}\n")
            f.write(f"Thermal resolution: {thermal_frame.shape[1]}x{thermal_frame.shape[0]}\n")
        
        return True
    except Exception as e:
        print(f"Warning: Failed to save frame {frame_idx}: {e}")
        return False


# ==================================================================
# MAIN STRESS TEST
# ==================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--duration", type=int, default=1200,
                         help="Test duration in seconds (default 1200 = 20 min)")
    parser.add_argument("--mode", choices=["flight", "maxrate"], default="flight",
                         help="flight = pace captures at the real trigger interval; "
                              "maxrate = capture as fast as possible (worst-case load)")
    parser.add_argument("--no-stream", action="store_true",
                         help="Disable HTTP video streaming")
    parser.add_argument("--no-save", action="store_true",
                         help="Disable saving frame pairs to disk")
    parser.add_argument("--save-interval", type=int, default=10,
                         help="Save every N frames (0=all, default=10)")
    parser.add_argument("--save-anomalies-only", action="store_true",
                         help="Only save frames with anomalies")
    parser.add_argument("--port", type=int, default=8080,
                         help="HTTP streaming port (default: 8080)")
    args = parser.parse_args()

    # Override config with command line args
    enable_stream = not args.no_stream
    enable_save = not args.no_save
    save_interval = args.save_interval
    save_anomalies_only = args.save_anomalies_only
    STREAM_PORT = args.port

    # Create log directory
    run_dir = Path("logs") / datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir.mkdir(parents=True, exist_ok=True)
    
    # Create directory for saved frames
    if enable_save:
        pairs_dir = run_dir / "saved_pairs"
        pairs_dir.mkdir(parents=True, exist_ok=True)
        print(f"Saving frame pairs to: {pairs_dir}/")
    
    print(f"Logging to {run_dir}/  (mode={args.mode}, duration={args.duration}s)")
    print(f"HTTP Stream: {'ON' if enable_stream else 'OFF'} (port {STREAM_PORT})")
    print(f"Saving: {'ON' if enable_save else 'OFF'} (interval={save_interval}, anomalies_only={save_anomalies_only})")

    # Start streaming server if enabled
    stream_server = None
    if enable_stream:
        stream_server = StreamingServer(port=STREAM_PORT)
        stream_server.start()
        print(f"\nTo view stream from Mac:")
        print(f"   1. In Mac terminal: ssh -L {STREAM_PORT}:localhost:{STREAM_PORT} pi5@<pi-ip>")
        print(f"   2. Open browser: http://localhost:{STREAM_PORT}")
        print()

    # Start system monitor
    stop_event = threading.Event()
    monitor = SystemMonitor(run_dir / "system_metrics.csv", SYSTEM_SAMPLE_INTERVAL_S, stop_event)
    monitor.start()

    # Create queues
    rgb_queue = queue.Queue(maxsize=16)
    thermal_queue = queue.Queue(maxsize=16)

    # Start capture threads
    rgb_capture = CaptureThread(RGB_DEVICE, rgb_queue, format_yuyv=False, width=640, height=480)
    thermal_capture = CaptureThread(THERMAL_DEVICE, thermal_queue, format_yuyv=True, width=256, height=192)
    rgb_capture.start()
    thermal_capture.start()
    print("Capture threads started.\n")

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

    # Open pipeline log
    pipeline_log_path = run_dir / "pipeline_metrics.csv"
    with open(pipeline_log_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "elapsed_s", "frame_idx", 
            "rgb_total_ms", "rgb_clahe_ms", "rgb_yolo_ms", "rgb_classify_ms",
            "n_panels", "n_anomalies",
            "thermal_label", "thermal_conf", "thermal_ms",
            "loop_total_ms", "instant_fps"
        ])

        start_time = time.time()
        frame_idx = 0
        frame_pairs_processed = 0
        saved_count = 0
        stream_fps_counter = 0
        stream_fps_time = time.time()
        
        print(f"\nStarting stress test (mode={args.mode})...")
        print(f"   Trigger interval: {FLIGHT_TRIGGER_INTERVAL_S}s")
        print("   Press Ctrl+C to stop.\n")
        
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
                
                # Process Thermal
                th_label, th_conf, th_anomaly_prob, th_ms = thermal_pipeline.predict(thermal_frame)
                rgb_result['timings']['thermal_ms'] = th_ms
                
                # Count anomalies
                n_anomalies = sum(1 for p in rgb_result["panel_results"] if p["label"] == "Anomaly")
                has_anomaly = (n_anomalies > 0 or th_label == "ANOMALY")
                
                # Calculate timings
                loop_total_ms = (time.time() - loop_t0) * 1000
                instant_fps = 1000.0 / loop_total_ms if loop_total_ms > 0 else float("inf")
                
                # Log to CSV
                writer.writerow([
                    f"{elapsed:.2f}", frame_idx,
                    f"{rgb_result['timings']['total']:.1f}",
                    f"{rgb_result['timings']['clahe']:.1f}",
                    f"{rgb_result['timings']['yolo']:.1f}",
                    f"{rgb_result['timings']['classify']:.1f}",
                    len(rgb_result['panel_results']),
                    n_anomalies,
                    th_label,
                    f"{th_conf:.4f}",
                    f"{th_ms:.1f}",
                    f"{loop_total_ms:.1f}",
                    f"{instant_fps:.2f}"
                ])
                f.flush()
                
                # Create display frame for streaming/saving
                combined_display = create_simple_display(rgb_frame, thermal_frame)
                
                # Add minimal info at bottom
                info_text = f"Frame: {frame_idx} | Elapsed: {elapsed:.1f}s | Mode: {args.mode} | Panels: {len(rgb_result['panel_results'])} | Anomalies: {n_anomalies} | Thermal: {th_label}"
                cv2.putText(combined_display, info_text, (10, combined_display.shape[0] - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                
                # Update streaming server with the frame
                if enable_stream and stream_server:
                    stream_server.update_frame(combined_display)
                
                # Save frame pairs if enabled
                if enable_save:
                    should_save = False
                    if save_anomalies_only and has_anomaly:
                        should_save = True
                    elif not save_anomalies_only:
                        if save_interval == 0:
                            should_save = True
                        elif frame_idx % save_interval == 0:
                            should_save = True
                    
                    if should_save:
                        if save_frame_pair(pairs_dir, frame_idx, rgb_frame, thermal_frame, elapsed):
                            saved_count += 1
                
                # Print progress every 50 frames
                if frame_idx % 50 == 0:
                    print(f"t={elapsed:6.1f}s  frame={frame_idx:5d}  "
                          f"fps={instant_fps:5.2f}  loop={loop_total_ms:6.1f}ms  "
                          f"panels={len(rgb_result['panel_results']):3d}  "
                          f"anomalies={n_anomalies:3d}  thermal={th_label}  "
                          f"saved={saved_count}")
                
                frame_idx += 1
                frame_pairs_processed += 1
                
                # Flight mode pacing - EXACTLY 1.875 seconds
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
            stop_event.set()
            monitor.join(timeout=5)
            
            if enable_stream and stream_server:
                stream_server.stop()
            
            print(f"\nStress test completed.")
            print(f"   Processed {frame_pairs_processed} frame pairs")
            print(f"   Saved {saved_count} frame pairs to disk")
            print(f"   Logs saved to: {run_dir}/")
            print(f"   System metrics: {run_dir}/system_metrics.csv")
            print(f"   Pipeline metrics: {run_dir}/pipeline_metrics.csv")
            if enable_save:
                print(f"   Saved frames: {run_dir}/saved_pairs/")


if __name__ == "__main__":
    main()
