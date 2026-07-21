"""
RGB + Thermal PARALLEL Capture + Sequential Processing
Optimized for Raspberry Pi 5
"""

import cv2
import numpy as np
import time
import csv
from pathlib import Path
from datetime import datetime
import threading
import queue
import signal
import sys

# ====================== RGB PIPELINE ======================
import onnxruntime as ort
from ultralytics import YOLO

YOLO_MODEL_PATH = "/home/pi5/Documents/panel_detection2/best_ncnn_model"
CLASSIFIER_ONNX_PATH = "/home/pi5/Documents/binary_class/onnx_128/classifier_opt.onnx"

YOLO_CONF = 0.35
YOLO_IOU = 0.45
YOLO_IMGSZ = 640
CLASSIFIER_IMGSZ = 128

ANOMALY_THRESHOLD = 0.60
UNCERTAIN_BAND = 0.10

_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(1, 3, 1, 1)
_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(1, 3, 1, 1)
_CLAHE_OBJ = cv2.createCLAHE(clipLimit=1.5, tileGridSize=(4, 4))


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
        self.classifier = ort.InferenceSession(CLASSIFIER_ONNX_PATH, sess_options=sess_options,
                                               providers=["CPUExecutionProvider"])
        self.cls_input_name = self.classifier.get_inputs()[0].name
        print("RGB models loaded.")

    def process(self, rgb_frame):
        t0 = time.perf_counter()
        clahe_img = apply_clahe_fast(rgb_frame)
        t_clahe = (time.perf_counter() - t0) * 1000

        t1 = time.perf_counter()
        results = self.yolo.predict(clahe_img, imgsz=YOLO_IMGSZ, conf=YOLO_CONF, iou=YOLO_IOU,
                                    half=False, augment=False, verbose=False)
        t_yolo = (time.perf_counter() - t1) * 1000

        t2 = time.perf_counter()
        crops = []
        h, w = clahe_img.shape[:2]
        panel_results = []

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
                        "confidence": float(probs.max())
                    })

        t_classify = (time.perf_counter() - t2) * 1000
        total_ms = (time.perf_counter() - t0) * 1000

        return {
            "clahe_img": clahe_img,
            "panel_results": panel_results,
            "timings": {"clahe": t_clahe, "yolo": t_yolo, "classify": t_classify, "total": total_ms}
        }


# ====================== THERMAL DETECTOR ======================
class ThermalPanelDetector:
    def __init__(self, model_path='thermal_binary_mobilenetv2_f16.tflite', threshold=0.5):
        try:
            from ai_edge_litert.interpreter import Interpreter
        except ImportError:
            try:
                from tflite_runtime.interpreter import Interpreter
            except ImportError:
                from tensorflow.lite.python.interpreter import Interpreter
        self.interpreter = Interpreter(model_path=str(model_path))
        self.interpreter.allocate_tensors()
        self.threshold = threshold
        self.input_details = self.interpreter.get_input_details()[0]
        self.output_details = self.interpreter.get_output_details()[0]
        self.input_size = 224

    def preprocess(self, image):
        if len(image.shape) == 3 and image.shape[2] == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        resized = cv2.resize(image, (self.input_size, self.input_size))
        normalized = (resized.astype(np.float32) / 127.5) - 1.0
        return np.expand_dims(normalized, axis=0)

    def predict(self, image):
        input_tensor = self.preprocess(image)
        self.interpreter.set_tensor(self.input_details['index'], input_tensor)
        self.interpreter.invoke()
        prob = float(self.interpreter.get_tensor(self.output_details['index'])[0][0])
        if prob < self.threshold:
            return "ANOMALY", 1 - prob, prob
        return "NO_ANOMALY", prob, prob


# ====================== MAIN INTEGRATOR ======================
class RGBThermalParallelIntegrator:
    def __init__(self, rgb_pipeline, thermal_detector, target_captures=20, target_fps=4):
        self.rgb_pipe = rgb_pipeline
        self.thermal_det = thermal_detector
        self.target_captures = target_captures
        self.interval = 1.0 / target_fps

        self.base_dir = Path(__file__).parent
        self.output_dir = self.base_dir / 'rgb_thermal_pairs'
        self.output_dir.mkdir(exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.csv_path = self.base_dir / f'rgb_thermal_log_{timestamp}.csv'
        self.setup_csv()

        self.rgb_queue = queue.Queue(maxsize=8)
        self.thermal_queue = queue.Queue(maxsize=8)
        self.pair_queue = queue.Queue(maxsize=8)

        self.running = True
        self.capture_count = 0

    def setup_csv(self):
        with open(self.csv_path, 'w', newline='') as f:
            csv.writer(f).writerow([
                'capture_id', 'timestamp', 'rgb_anomalies', 'thermal_label', 'thermal_conf',
                'rgb_total_ms', 'thermal_ms', 'combined_ms'
            ])

    def rgb_capture_thread(self, cap_rgb):
        while self.running:
            ret, frame = cap_rgb.read()
            if ret:
                self.rgb_queue.put((time.time(), frame))
            time.sleep(0.001)

    def thermal_capture_thread(self, cap_thermal):
        while self.running:
            ret, frame = cap_thermal.read()
            if ret:
                bgr = cv2.cvtColor(frame, cv2.COLOR_YUV2BGR_YUYV)
                gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
                colored = cv2.applyColorMap(gray, cv2.COLORMAP_INFERNO)
                self.thermal_queue.put((time.time(), colored))
            time.sleep(0.001)

    def pair_frames(self):
        """Pair RGB and Thermal frames by closest timestamp"""
        while self.running:
            try:
                rgb_ts, rgb_frame = self.rgb_queue.get(timeout=0.5)
                thermal_ts, thermal_frame = self.thermal_queue.get(timeout=0.5)

                # Simple sync: use the later timestamp
                pair_ts = max(rgb_ts, thermal_ts)
                self.pair_queue.put((pair_ts, rgb_frame, thermal_frame))
            except queue.Empty:
                continue
            except Exception as e:
                print(f"Pairing error: {e}")

    def process_pair(self, rgb_frame, thermal_frame):
        t_start = time.perf_counter()

        rgb_result = self.rgb_pipe.process(rgb_frame)

        t_th = time.perf_counter()
        thermal_label, thermal_conf, _ = self.thermal_det.predict(thermal_frame)
        thermal_ms = (time.perf_counter() - t_th) * 1000

        combined_ms = (time.perf_counter() - t_start) * 1000
        return rgb_result, thermal_label, thermal_conf, thermal_ms, combined_ms

    def run(self):
        print("Initializing cameras...")
        # Adjust indices as needed (run ls /dev/video* to check)
        cap_rgb = cv2.VideoCapture(2, cv2.CAP_V4L2)      # RGB camera
        cap_thermal = cv2.VideoCapture(0, cv2.CAP_V4L2)  # Thermal

        cap_rgb.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap_rgb.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        cap_thermal.set(cv2.CAP_PROP_CONVERT_RGB, 0)
        cap_thermal.set(cv2.CAP_PROP_FRAME_WIDTH, 256)
        cap_thermal.set(cv2.CAP_PROP_FRAME_HEIGHT, 192)

        # Start capture threads
        threading.Thread(target=self.rgb_capture_thread, args=(cap_rgb,), daemon=True).start()
        threading.Thread(target=self.thermal_capture_thread, args=(cap_thermal,), daemon=True).start()
        threading.Thread(target=self.pair_frames, daemon=True).start()

        print(f"Starting parallel capture at ~{1/self.interval:.1f} FPS...")

        def signal_handler(sig, frame):
            self.running = False
        signal.signal(signal.SIGINT, signal_handler)

        while self.running and self.capture_count < self.target_captures:
            try:
                _, rgb_frame, thermal_frame = self.pair_queue.get(timeout=1.0)
            except queue.Empty:
                continue

            rgb_result, th_label, th_conf, th_ms, comb_ms = self.process_pair(rgb_frame, thermal_frame)
            self.capture_count += 1

            # Logging
            anomalies = sum(1 for p in rgb_result["panel_results"] if p["label"] == "Anomaly")
            with open(self.csv_path, 'a', newline='') as f:
                csv.writer(f).writerow([
                    self.capture_count, datetime.now().isoformat(), anomalies,
                    th_label, f"{th_conf:.4f}", f"{rgb_result['timings']['total']:.1f}",
                    f"{th_ms:.1f}", f"{comb_ms:.1f}"
                ])

            print(f"\n[Capture {self.capture_count:03d}] "
                  f"RGB: {len(rgb_result['panel_results'])} panels | {anomalies} anomalies | "
                  f"{rgb_result['timings']['total']:.1f}ms | "
                  f"Thermal: {th_label} ({th_conf:.1%}) | {th_ms:.1f}ms | "
                  f"Total: {comb_ms:.1f}ms")

            # Save interesting pairs
            if anomalies > 0 or th_label == "ANOMALY":
                pair_dir = self.output_dir / f"cap_{self.capture_count:03d}"
                pair_dir.mkdir(exist_ok=True)
                cv2.imwrite(str(pair_dir / "rgb.jpg"), rgb_result["clahe_img"])
                cv2.imwrite(str(pair_dir / "thermal.jpg"), thermal_frame)

        cap_rgb.release()
        cap_thermal.release()
        print(f"\n✅ Completed {self.capture_count} synchronized pairs.")
        print(f"Log saved to: {self.csv_path}")


if __name__ == "__main__":
    rgb_pipe = RGBPipeline()
    thermal_det = ThermalPanelDetector(threshold=0.5)

    integrator = RGBThermalParallelIntegrator(rgb_pipe, thermal_det, target_captures=30, target_fps=4)
    integrator.run()
