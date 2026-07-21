"""
Final Clean Version with Detailed Logging
"""

import cv2
import numpy as np
import time
import os
import glob
from pathlib import Path
from dataclasses import dataclass
from typing import List
import threading
import queue
from concurrent.futures import ThreadPoolExecutor, as_completed

# ========================= CONFIG =========================
YOLO_MODEL_PATH = "/home/pi5/panel_detection2/best_ncnn_model"
CLASSIFIER_ONNX_PATH = "/home/pi5/binary_class/ncnn_export/classifier.onnx"
IMAGE_FOLDER = "/home/pi5/binary_class/test_images"
OUTPUT_FOLDER = "/home/pi5/panel_detection2/pipeline_results"

YOLO_CONF = 0.35
YOLO_IMGSZ = 512
CLASSIFIER_IMGSZ = 224
FRAME_SKIP = 4

CLASSES = ["Healthy", "Anomaly"]

_CLAHE = cv2.createCLAHE(clipLimit=1.5, tileGridSize=(4, 4))

@dataclass
class FrameData:
    frame_idx: int
    path: str
    raw: np.ndarray
    clahe: np.ndarray = None
    detections: list = None
    results: List[dict] = None
    t_clahe: float = 0.0
    t_yolo: float = 0.0
    t_classify: float = 0.0


def load_and_resize(path: str, size=(512, 512)):
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None: return None
    return cv2.resize(img, size, interpolation=cv2.INTER_LINEAR)


def apply_clahe(img: np.ndarray) -> np.ndarray:
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    l = _CLAHE.apply(l)
    return cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_LAB2BGR)


def extract_crops(img: np.ndarray, yolo_results) -> List[np.ndarray]:
    crops = []
    if not yolo_results: return crops
    h, w = img.shape[:2]
    for r in yolo_results:
        if getattr(r, 'obb', None) is None: continue
        for box in r.obb.xyxyxyxy.cpu().numpy():
            pts = box.reshape(4, 2)
            x1 = max(0, int(pts[:,0].min()))
            y1 = max(0, int(pts[:,1].min()))
            x2 = min(w, int(pts[:,0].max()))
            y2 = min(h, int(pts[:,1].max()))
            if x2 > x1 and y2 > y1:
                crops.append(img[y1:y2, x1:x2])
    return crops


def preprocess_crop(crop: np.ndarray, size=224):
    crop = cv2.resize(crop, (size, size), interpolation=cv2.INTER_LINEAR)
    crop = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std  = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    crop = (crop - mean) / std
    return crop.transpose(2, 0, 1)[np.newaxis]


def softmax(x):
    e = np.exp(x - x.max(axis=1, keepdims=True))
    return e / e.sum(axis=1, keepdims=True)


class Pipeline:
    def __init__(self):
        os.makedirs(OUTPUT_FOLDER, exist_ok=True)
        
        print("Loading YOLO...")
        from ultralytics import YOLO
        self.yolo = YOLO(YOLO_MODEL_PATH, task="obb")
        
        print("Loading ONNX Classifier...")
        import onnxruntime as ort
        sess_options = ort.SessionOptions()
        sess_options.intra_op_num_threads = 3
        self.classifier = ort.InferenceSession(
            CLASSIFIER_ONNX_PATH, sess_options=sess_options, providers=["CPUExecutionProvider"]
        )
        self.cls_input_name = self.classifier.get_inputs()[0].name

        self._last_detections = None
        self.q_detect = queue.Queue(maxsize=20)
        self.q_done = queue.Queue(maxsize=20)

    def clahe_stage(self, fd: FrameData):
        t0 = time.perf_counter()
        fd.clahe = apply_clahe(fd.raw)
        fd.t_clahe = (time.perf_counter() - t0) * 1000
        return fd

    def detect_worker(self):
        while True:
            try:
                fd = self.q_detect.get(timeout=3.0)
            except queue.Empty:
                continue
            if fd is None: break

            t0 = time.perf_counter()
            if fd.frame_idx % FRAME_SKIP == 0:
                results = self.yolo.predict(fd.clahe, imgsz=YOLO_IMGSZ, conf=YOLO_CONF, verbose=False)
                self._last_detections = results
            else:
                results = self._last_detections

            fd.detections = results
            fd.t_yolo = (time.perf_counter() - t0) * 1000
            self.q_done.put(fd)

    def classify_worker(self):
        while True:
            try:
                fd = self.q_done.get(timeout=3.0)
            except queue.Empty:
                continue
            if fd is None: break

            t0 = time.perf_counter()
            crops = extract_crops(fd.clahe, fd.detections)
            panel_results = []

            if crops:
                batch = np.concatenate([preprocess_crop(c) for c in crops], axis=0)
                logits = self.classifier.run(None, {self.cls_input_name: batch})[0]
                probs = softmax(logits)

                for i, p in enumerate(probs):
                    pred = int(p.argmax())
                    panel_results.append({
                        "panel_id": i+1,
                        "label": CLASSES[pred],
                        "confidence": float(p[pred])
                    })

            fd.results = panel_results
            fd.t_classify = (time.perf_counter() - t0) * 1000

            total_ms = fd.t_clahe + fd.t_yolo + fd.t_classify
            fps = 1000 / total_ms if total_ms > 0 else 0

            status = "⚠️ ANOMALY" if any(r["label"] == "Anomaly" for r in panel_results) else "✅ Healthy"
            print(f"[{fd.frame_idx+1:02d}] {Path(fd.path).name:<20} | "
                  f"{len(panel_results)} panels | {status} | {fps:5.1f} FPS "
                  f"(C:{fd.t_clahe:4.1f} Y:{fd.t_yolo:4.1f} Cls:{fd.t_classify:4.1f}ms)")

            self._save_result(fd)

    def run(self, image_paths):
        frame_datas = [FrameData(idx, p, load_and_resize(p, (YOLO_IMGSZ, YOLO_IMGSZ))) 
                      for idx, p in enumerate(image_paths) if load_and_resize(p) is not None]

        print(f"Preloaded {len(frame_datas)} images. Warming up...\n")

        # Warmup
        self.yolo.predict(np.zeros((YOLO_IMGSZ, YOLO_IMGSZ, 3), np.uint8), imgsz=YOLO_IMGSZ, verbose=False)
        dummy = np.zeros((1, 3, CLASSIFIER_IMGSZ, CLASSIFIER_IMGSZ), np.float32)
        self.classifier.run(None, {self.cls_input_name: dummy})

        threading.Thread(target=self.detect_worker, daemon=True).start()
        threading.Thread(target=self.classify_worker, daemon=True).start()

        t_start = time.perf_counter()

        with ThreadPoolExecutor(max_workers=4) as executor:
            futures = [executor.submit(self.clahe_stage, fd) for fd in frame_datas]
            for future in as_completed(futures):
                self.q_detect.put(future.result())

        self.q_detect.put(None)
        self.q_done.put(None)

        elapsed = time.perf_counter() - t_start
        print(f"\n✅ Pipeline finished in {elapsed:.2f} seconds")


    def _save_result(self, fd):
        cv2.imwrite(os.path.join(OUTPUT_FOLDER, Path(fd.path).name), fd.clahe)


# ========================= MAIN =========================
if __name__ == "__main__":
    cv2.setNumThreads(4)
    
    image_paths = sorted(glob.glob(os.path.join(IMAGE_FOLDER, "*.[jp][pn]g")))
    print(f"Found {len(image_paths)} images\n")
    
    Pipeline().run(image_paths)
