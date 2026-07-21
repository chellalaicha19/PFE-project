"""
RGB + Thermal PARALLEL Simulation
Processes paired images from test_rgb+thermal/ folder
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
    batch = batch / 255.0 - _MEAN
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
        return {
            "clahe_img": clahe_img,
            "panel_results": panel_results,
            "timings": {"clahe": t_clahe, "yolo": t_yolo, "classify": t_classify, "total": (time.perf_counter()-t0)*1000}
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


# ====================== MAIN PARALLEL INTEGRATOR ======================
class RGBThermalParallelSimulator:
    def __init__(self, rgb_pipeline, thermal_detector, test_folder="test_rgb+thermal"):
        self.rgb_pipe = rgb_pipeline
        self.thermal_det = thermal_detector
        self.test_folder = Path(test_folder)
        self.output_dir = Path("rgb_thermal_results")
        self.output_dir.mkdir(exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.csv_path = self.output_dir / f'simulation_log_{timestamp}.csv'
        self.setup_csv()

    def setup_csv(self):
        with open(self.csv_path, 'w', newline='') as f:
            csv.writer(f).writerow([
                'pair_id', 'timestamp', 'rgb_anomalies', 'thermal_label', 'thermal_conf',
                'rgb_total_ms', 'thermal_ms', 'combined_wall_ms'
            ])

    def load_pairs(self):
        pairs = []
        rgb_files = sorted(self.test_folder.glob("*_rgb.png"))
        for rgb_path in rgb_files:
            num = rgb_path.stem.replace("_rgb", "")
            thermal_path = self.test_folder / f"{num}_thermal.png"
            if thermal_path.exists():
                pairs.append((rgb_path, thermal_path))
        return pairs

    def process_pair_parallel(self, rgb_path, thermal_path, pair_id):
        # Load images
        rgb_frame = cv2.imread(str(rgb_path))
        thermal_frame = cv2.imread(str(thermal_path))

        if rgb_frame is None or thermal_frame is None:
            print(f"❌ Failed to load pair {pair_id}")
            return None

        # Prepare thermal colored image
        gray = cv2.cvtColor(thermal_frame, cv2.COLOR_BGR2GRAY)
        thermal_colored = cv2.applyColorMap(gray, cv2.COLORMAP_INFERNO)

        # Parallel processing using threads
        rgb_result = None
        thermal_result = None
        rgb_done = threading.Event()
        thermal_done = threading.Event()

        def run_rgb():
            nonlocal rgb_result
            rgb_result = self.rgb_pipe.process(rgb_frame)
            rgb_done.set()

        def run_thermal():
            nonlocal thermal_result
            t0 = time.perf_counter()
            label, conf, _ = self.thermal_det.predict(thermal_colored)
            thermal_result = {"label": label, "conf": conf, "time_ms": (time.perf_counter() - t0) * 1000}
            thermal_done.set()

        t_start = time.perf_counter()

        # Start both pipelines near-simultaneously
        t1 = threading.Thread(target=run_rgb)
        t2 = threading.Thread(target=run_thermal)
        t1.start()
        t2.start()

        t1.join()
        t2.join()

        combined_ms = (time.perf_counter() - t_start) * 1000

        # Log
        anomalies = sum(1 for p in rgb_result["panel_results"] if p["label"] == "Anomaly")
        with open(self.csv_path, 'a', newline='') as f:
            csv.writer(f).writerow([
                pair_id,
                datetime.now().isoformat(),
                anomalies,
                thermal_result["label"],
                f"{thermal_result['conf']:.4f}",
                f"{rgb_result['timings']['total']:.1f}",
                f"{thermal_result['time_ms']:.1f}",
                f"{combined_ms:.1f}"
            ])

        print(f"\n[Pair {pair_id:02d}]")
        print(f"RGB     → {len(rgb_result['panel_results'])} panels | {anomalies} anomalies | {rgb_result['timings']['total']:.1f}ms")
        print(f"Thermal → {thermal_result['label']} ({thermal_result['conf']:.1%}) | {thermal_result['time_ms']:.1f}ms")
        print(f"**Combined wall time: {combined_ms:.1f}ms**")

        # Save visualization if anomaly
        if anomalies > 0 or thermal_result["label"] == "ANOMALY":
            save_dir = self.output_dir / f"pair_{pair_id:02d}"
            save_dir.mkdir(exist_ok=True)
            cv2.imwrite(str(save_dir / "rgb_processed.jpg"), rgb_result["clahe_img"])
            cv2.imwrite(str(save_dir / "thermal.jpg"), thermal_colored)

        return combined_ms

    def run(self):
        pairs = self.load_pairs()
        print(f"Found {len(pairs)} RGB-Thermal pairs to process.")

        if not pairs:
            print("No pairs found!")
            return

        total_start = time.perf_counter()
        combined_times = []

        for i, (rgb_p, th_p) in enumerate(pairs, 1):
            combined = self.process_pair_parallel(rgb_p, th_p, i)
            if combined:
                combined_times.append(combined)

        total_time = time.perf_counter() - total_start
        avg_combined = sum(combined_times)/len(combined_times) if combined_times else 0

        print("\n" + "="*70)
        print("SIMULATION SUMMARY")
        print("="*70)
        print(f"Pairs processed       : {len(pairs)}")
        print(f"Total wall time       : {total_time:.2f}s")
        print(f"Avg combined latency  : {avg_combined:.1f} ms per pair")
        print(f"Results saved to      : {self.output_dir}")
        print(f"Log file              : {self.csv_path}")
        print("="*70)


if __name__ == "__main__":
    rgb_pipe = RGBPipeline()
    thermal_det = ThermalPanelDetector(threshold=0.5)

    simulator = RGBThermalParallelSimulator(rgb_pipe, thermal_det, test_folder="test_rgb+thermal")
    simulator.run()