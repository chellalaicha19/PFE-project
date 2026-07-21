"""
Optimized Parallel Pipeline for Pi 5
- Multiprocessing CLAHE
- Threaded YOLO + Classifier
- Target: 15+ FPS
"""

import cv2
import numpy as np
import time
import os
import glob
from pathlib import Path
from dataclasses import dataclass
from typing import List, Optional
import multiprocessing as mp
import threading
import queue

# ========================= CONFIG =========================
YOLO_MODEL_PATH = "/home/pi5/panel_detection2/best_ncnn_model"
CLASSIFIER_ONNX_PATH = "/home/pi5/binary_class/ncnn_export/classifier.onnx"
IMAGE_FOLDER = "/home/pi5/binary_class/test_images"
OUTPUT_FOLDER = "/home/pi5/panel_detection2/pipeline_results"

YOLO_CONF = 0.35
YOLO_IMGSZ = 640          # Try 512 if you can accept small accuracy drop
CLASSIFIER_IMGSZ = 224    # Try 192 if needed

FRAME_SKIP = 1            # Increased from 2
QUEUE_SIZE = 6

CLASSES = ["Healthy", "Anomaly"]

# Faster CLAHE
_CLAHE = cv2.createCLAHE(clipLimit=1.8, tileGridSize=(4, 4))

# ====================== DATA CLASS ======================
@dataclass
class FrameData:
    frame_idx: int
    path: str
    raw: np.ndarray
    clahe: Optional[np.ndarray] = None
    detections: Optional[list] = None
    results: List[dict] = None
    t_clahe: float = 0.0
    t_yolo: float = 0.0
    t_classify: float = 0.0


# ====================== HELPERS ======================
def load_and_resize(path: str, size=(640, 640)):
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        return None
    return cv2.resize(img, size, interpolation=cv2.INTER_LINEAR)


def apply_clahe(img: np.ndarray) -> np.ndarray:
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    l = _CLAHE.apply(l)
    return cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_LAB2BGR)


def extract_crops(img: np.ndarray, yolo_results) -> List[np.ndarray]:
    crops = []
    h, w = img.shape[:2]
    if not yolo_results or not hasattr(yolo_results[0], 'obb'):
        return crops

    for r in yolo_results:
        if r.obb is None:
            continue
        for box in r.obb.xyxyxyxy.cpu().numpy():
            pts = box.reshape(4, 2)
            x1, y1 = max(0, int(pts[:, 0].min())), max(0, int(pts[:, 1].min()))
            x2, y2 = min(w, int(pts[:, 0].max())), min(h, int(pts[:, 1].max()))
            if x2 > x1 and y2 > y1:
                crops.append(img[y1:y2, x1:x2])
    return crops


def preprocess_crop(crop: np.ndarray, size: int = 224) -> np.ndarray:
    crop = cv2.resize(crop, (size, size), interpolation=cv2.INTER_LINEAR)
    crop = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    crop = (crop - mean) / std
    return crop.transpose(2, 0, 1)[np.newaxis]


def softmax(x):
    e = np.exp(x - x.max(axis=1, keepdims=True))
    return e / e.sum(axis=1, keepdims=True)


# ====================== PIPELINE ======================
class OptimizedPipeline:
    def __init__(self):
        os.makedirs(OUTPUT_FOLDER, exist_ok=True)
        
        print("Loading YOLO (NCNN)...")
        from ultralytics import YOLO
        self.yolo = YOLO(YOLO_MODEL_PATH, task="obb")
        
        print("Loading Classifier (ONNX)...")
        import onnxruntime as ort
        sess_options = ort.SessionOptions()
        sess_options.intra_op_num_threads = 3
        self.classifier = ort.InferenceSession(
            CLASSIFIER_ONNX_PATH,
            sess_options=sess_options,
            providers=["CPUExecutionProvider"]
        )
        self.cls_input_name = self.classifier.get_inputs()[0].name

        self._last_detections = None
        self._stop = threading.Event()
        
        # Queues
        self.q_clahe = mp.Queue(maxsize=QUEUE_SIZE)   # from pool → detect
        self.q_detect = queue.Queue(maxsize=QUEUE_SIZE)
        self.q_done = queue.Queue(maxsize=QUEUE_SIZE)

    def clahe_pool_worker(self, frame_data_list):
        """Used by multiprocessing pool"""
        for fd in frame_data_list:
            t0 = time.perf_counter()
            fd.clahe = apply_clahe(fd.raw)
            fd.t_clahe = (time.perf_counter() - t0) * 1000
            self.q_clahe.put(fd)
        return True

    def detect_worker(self):
        while not self._stop.is_set():
            try:
                fd: FrameData = self.q_clahe.get(timeout=1.0)
            except:
                continue

            t0 = time.perf_counter()
            if fd.frame_idx % FRAME_SKIP == 0:
                results = self.yolo.predict(
                    fd.clahe, imgsz=YOLO_IMGSZ, conf=YOLO_CONF,
                    verbose=False, device='cpu'
                )
                self._last_detections = results
            else:
                results = self._last_detections

            fd.detections = results
            fd.t_yolo = (time.perf_counter() - t0) * 1000
            self.q_detect.put(fd)

    def classify_worker(self):
        while not self._stop.is_set():
            try:
                fd: FrameData = self.q_detect.get(timeout=1.0)
            except:
                continue

            t0 = time.perf_counter()
            crops = extract_crops(fd.clahe, fd.detections)

            panel_results = []
            if crops:
                batch = np.concatenate([preprocess_crop(c, CLASSIFIER_IMGSZ) for c in crops], axis=0)
                logits = self.classifier.run(None, {self.cls_input_name: batch})[0]
                probs = softmax(logits)

                for i, p in enumerate(probs):
                    pred = int(p.argmax())
                    panel_results.append({
                        "panel_id": i + 1,
                        "label": CLASSES[pred],
                        "confidence": float(p.max()),
                        "healthy_prob": float(p[0]),
                        "anomaly_prob": float(p[1]),
                    })

            fd.results = panel_results
            fd.t_classify = (time.perf_counter() - t0) * 1000
            self.q_done.put(fd)

    def run(self, image_paths: List[str]):
        # Preload all images
        print(f"Preloading {len(image_paths)} images...")
        frame_datas = []
        for idx, path in enumerate(image_paths):
            raw = load_and_resize(path, (YOLO_IMGSZ, YOLO_IMGSZ))
            if raw is not None:
                frame_datas.append(FrameData(frame_idx=idx, path=path, raw=raw))

        print("Warming up models...")
        self.yolo.predict(np.zeros((YOLO_IMGSZ, YOLO_IMGSZ, 3), dtype=np.uint8),
                         imgsz=YOLO_IMGSZ, verbose=False)
        dummy = np.zeros((1, 3, CLASSIFIER_IMGSZ, CLASSIFIER_IMGSZ), dtype=np.float32)
        self.classifier.run(None, {self.cls_input_name: dummy})

        # Start workers
        detect_thread = threading.Thread(target=self.detect_worker, daemon=True)
        classify_thread = threading.Thread(target=self.classify_worker, daemon=True)
        detect_thread.start()
        classify_thread.start()

        # Multiprocessing Pool for CLAHE
        num_processes = max(2, mp.cpu_count() - 2)   # Leave cores for YOLO + classifier
        pool = mp.Pool(processes=num_processes)

        print(f"\nStarting pipeline with {num_processes} CLAHE processes...\n")
        t_start = time.perf_counter()

        # Feed in chunks to pool
        chunk_size = max(1, len(frame_datas) // (num_processes * 2))
        chunks = [frame_datas[i:i+chunk_size] for i in range(0, len(frame_datas), chunk_size)]

        for chunk in chunks:
            pool.apply_async(self.clahe_pool_worker, (chunk,))

        # Collect results
        collected = 0
        fps_log = []
        total_images = len(frame_datas)

        while collected < total_images:
            try:
                fd: FrameData = self.q_done.get(timeout=8.0)
            except queue.Empty:
                print("Pipeline timeout")
                break

            total_ms = fd.t_clahe + fd.t_yolo + fd.t_classify
            fps = 1000 / total_ms if total_ms > 0 else 0
            fps_log.append(fps)
            collected += 1

            # Minimal printing (remove for real deployment)
            anomalies = sum(1 for r in fd.results if r["label"] == "Anomaly")
            status = f"⚠️ {anomalies} ANOMALY" if anomalies else "✅ Healthy"
            print(f"[{collected:03d}/{total_images}] {Path(fd.path).name:<25} | "
                  f"{len(fd.results)} panels | {status} | {fps:5.1f} FPS")

            self._save_result(fd)

        pool.close()
        pool.join()
        self._stop.set()

        elapsed = time.perf_counter() - t_start
        print("\n" + "="*70)
        print("FINAL SUMMARY")
        print(f"Images processed : {collected}")
        print(f"Wall time        : {elapsed:.2f}s")
        print(f"Avg per-frame FPS: {sum(fps_log)/len(fps_log):.1f}")
        print(f"Throughput       : {collected/elapsed:.1f} img/s")
        print(f"Results saved to : {OUTPUT_FOLDER}")
        print("="*70)

    def _save_result(self, fd: FrameData):
        img = fd.clahe.copy()
        # ... (keep your existing drawing code)
        cv2.imwrite(os.path.join(OUTPUT_FOLDER, Path(fd.path).name), img)


# ========================= MAIN =========================
if __name__ == "__main__":
    cv2.setNumThreads(4)
    
    exts = ["*.jpg", "*.jpeg", "*.png"]
    image_paths = []
    for ext in exts:
        image_paths.extend(sorted(glob.glob(os.path.join(IMAGE_FOLDER, ext))))

    if not image_paths:
        print("No images found!")
        exit(1)

    OptimizedPipeline().run(image_paths)
