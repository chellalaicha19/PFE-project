"""
Full parallel pipeline: CLAHE → YOLO (OBB) → Classifier (ONNXRuntime)
With stability fixes applied - PROPER SHUTDOWN VERSION
"""

import cv2
import numpy as np
import threading
import queue
import time
import os
import glob
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import onnxruntime as ort
from ultralytics import YOLO

# ═══════════════════════════════════════════════════════════════
# CONFIGURATION
# ═══════════════════════════════════════════════════════════════
YOLO_MODEL_PATH      = "/home/pi5/panel_detection2/best_ncnn_model"
CLASSIFIER_ONNX_PATH = "/home/pi5/binary_class/onnx_mobileNet/classifier.onnx"
IMAGE_FOLDER         = "/home/pi5/binary_class/test_images"
OUTPUT_FOLDER        = "/home/pi5/panel_detection2/pipeline_results"

YOLO_CONF            = 0.35
YOLO_IOU             = 0.45
YOLO_IMGSZ           = 640
CLASSIFIER_IMGSZ     = 224

FRAME_SKIP           = 2
QUEUE_SIZE           = 4

CLASSES              = ["Healthy", "Anomaly"]

# Stability thresholds
ANOMALY_THRESHOLD    = 0.60
UNCERTAIN_BAND       = 0.10

# Fix 1: Define normalization constants ONCE (needed for batch preprocessing)
_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(1, 3, 1, 1)
_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(1, 3, 1, 1)

# Fix 2: Create CLAHE object ONCE at module level (not per image)
_CLAHE_OBJ = cv2.createCLAHE(clipLimit=1.5, tileGridSize=(4, 4))


@dataclass
class FrameData:
    frame_idx: int
    path: str
    raw: np.ndarray
    resized: Optional[np.ndarray] = None  # Resized to 640x640
    clahe: Optional[np.ndarray] = None   # CLAHE applied after resize
    detections: Optional[object] = None
    results: List[dict] = field(default_factory=list)
    t_load: float = 0.0
    t_resize: float = 0.0
    t_clahe: float = 0.0
    t_yolo: float = 0.0
    t_classify: float = 0.0


def load_image(path: str) -> Optional[np.ndarray]:
    """Load image at original size."""
    img = cv2.imread(path)
    if img is None:
        return None
    return img


def resize_for_yolo(img: np.ndarray, size=(640, 640)) -> np.ndarray:
    """Resize image to YOLO input size."""
    return cv2.resize(img, size)


def apply_clahe_fast(img: np.ndarray, clahe_obj) -> np.ndarray:
    """Optimized CLAHE - reuses existing CLAHE object on already resized image."""
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    # Apply CLAHE only to L channel (no need to split and merge)
    lab[:,:,0] = clahe_obj.apply(lab[:,:,0])
    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)


def classify_panel(prob_anomaly: float) -> Tuple[str, str]:
    """Stable classification with dead-zone."""
    if prob_anomaly >= ANOMALY_THRESHOLD:
        return "Anomaly", "🔴"
    elif prob_anomaly >= (ANOMALY_THRESHOLD - UNCERTAIN_BAND):
        return "Uncertain", "🟡"
    else:
        return "Healthy", "🟢"


def extract_crops(img: np.ndarray, yolo_results) -> List[np.ndarray]:
    """Extract crops from the CLAHE-processed resized image."""
    crops = []
    h, w = img.shape[:2]
    if not yolo_results:
        return crops

    for r in yolo_results:
        if r.obb is None:
            continue
        for box in r.obb.xyxyxyxy.cpu().numpy():
            pts = box.reshape(4, 2)
            x1 = max(0, int(pts[:, 0].min()))
            y1 = max(0, int(pts[:, 1].min()))
            x2 = min(w, int(pts[:, 0].max()))
            y2 = min(h, int(pts[:, 1].max()))
            if x2 > x1 and y2 > y1:
                crops.append(img[y1:y2, x1:x2])
    return crops


def preprocess_batch(crops: list, size: int = 128) -> np.ndarray:
    """Process all crops at once using vectorized ops — no Python loop overhead."""
    if not crops:
        return np.empty((0, 3, size, size), dtype=np.float32)
    
    n = len(crops)
    batch = np.empty((n, 3, size, size), dtype=np.float32)
    
    for i, crop in enumerate(crops):
        resized = cv2.resize(crop, (size, size))
        batch[i] = np.ascontiguousarray(resized[:, :, ::-1]).transpose(2, 0, 1)
    
    batch /= 255.0
    batch -= _MEAN
    batch /= _STD
    return batch


def softmax(x):
    e = np.exp(x - x.max())
    return e / e.sum()


class Pipeline:
    def __init__(self):
        os.makedirs(OUTPUT_FOLDER, exist_ok=True)

        print("Loading YOLO model...")
        self.yolo = YOLO(YOLO_MODEL_PATH, task="obb")

        print("Loading ONNX classifier...")
        sess_options = ort.SessionOptions()
        sess_options.intra_op_num_threads = 2
        sess_options.inter_op_num_threads = 1
        sess_options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        self.classifier = ort.InferenceSession(
            CLASSIFIER_ONNX_PATH,
            sess_options=sess_options,
            providers=["CPUExecutionProvider"]
        )
        self.cls_input_name = self.classifier.get_inputs()[0].name

        self.q_raw   = queue.Queue(maxsize=QUEUE_SIZE)
        self.q_resized = queue.Queue(maxsize=QUEUE_SIZE)
        self.q_done  = queue.Queue()

        self._stop = threading.Event()
        self.q_save = queue.Queue(maxsize=8)
        
        # Store CLAHE object as instance variable for reuse
        self.clahe_obj = cv2.createCLAHE(clipLimit=1.5, tileGridSize=(4, 4))

    def warmup(self):
        """Full pipeline warmup to initialize all buffers and caches."""
        print("=== Warming up pipeline ===")

        print("Warming up YOLO...")
        dummy = np.zeros((640, 640, 3), dtype=np.uint8)
        self.yolo.predict(
            dummy,
            imgsz=YOLO_IMGSZ,
            conf=YOLO_CONF,
            iou=YOLO_IOU,
            half=False,
            augment=False,
            verbose=False
        )

        print("Warming up Classifier...")
        dummy_crop = np.zeros((1, 3, CLASSIFIER_IMGSZ, CLASSIFIER_IMGSZ), dtype=np.float32)
        self.classifier.run(None, {self.cls_input_name: dummy_crop})
        
        # Warm up CLAHE with dummy image to initialize any internal state
        dummy_clahe = np.random.randint(100, 200, (640, 640, 3), dtype=np.uint8)
        apply_clahe_fast(dummy_clahe, self.clahe_obj)

        print("Pipeline warmup completed.\n")

    def check_yolo_determinism(self, test_img_path: str, runs: int = 3):
        """Check if YOLO is stable (important on NCNN/ARM)."""
        try:
            img = cv2.imread(test_img_path)
            if img is None:
                print("[WARNING] Could not load test image for determinism check")
                return
            
            img_resized = cv2.resize(img, (YOLO_IMGSZ, YOLO_IMGSZ))
            counts = []
            for _ in range(runs):
                r = self.yolo.predict(
                    img_resized,
                    imgsz=YOLO_IMGSZ,
                    conf=YOLO_CONF,
                    iou=YOLO_IOU,
                    half=False,
                    augment=False,
                    verbose=False
                )
                counts.append(len(r[0].boxes) if r and r[0].boxes is not None else 0)
            
            if len(set(counts)) > 1:
                print(f"[WARNING] YOLO is NON-DETERMINISTIC: {counts}")
            else:
                print(f"[OK] YOLO stable: {counts[0]} detections across {runs} runs")
        except Exception as e:
            print(f"[WARNING] Determinism check failed: {e}")
            
    def save_worker(self):
        """Save worker - processes saves from queue"""
        while True:
            try:
                item = self.q_save.get(timeout=1)
                if item is None:
                    break
                self._save_result(item)
                self.q_save.task_done()
            except queue.Empty:
                if self._stop.is_set() and self.q_save.empty():
                    break
                continue
            except Exception as e:
                print(f"Save worker error: {e}")
                continue

    def resize_worker(self):
        """Resize images to 640x640 (no CLAHE yet)."""
        while not self._stop.is_set():
            try:
                fd: FrameData = self.q_raw.get(timeout=0.5)
            except queue.Empty:
                continue
            
            # Only resize, no CLAHE
            t0 = time.perf_counter()
            fd.resized = resize_for_yolo(fd.raw, (YOLO_IMGSZ, YOLO_IMGSZ))
            fd.t_resize = (time.perf_counter() - t0) * 1000
            
            self.q_resized.put(fd)

    def clahe_worker(self):
        """Apply CLAHE to already resized images."""
        local_clahe = self.clahe_obj
        
        while not self._stop.is_set():
            try:
                fd: FrameData = self.q_resized.get(timeout=0.5)
            except queue.Empty:
                continue
            
            # Apply CLAHE to resized image (640x640)
            t0 = time.perf_counter()
            fd.clahe = apply_clahe_fast(fd.resized, local_clahe)
            fd.t_clahe = (time.perf_counter() - t0) * 1000
            
            self.q_done.put(fd)

    def detect_worker(self):
        while not self._stop.is_set():
            try:
                fd: FrameData = self.q_done.get(timeout=0.5)
            except queue.Empty:
                continue

            t0 = time.perf_counter()
            results = self.yolo.predict(
                fd.clahe,  # Use CLAHE-processed resized image
                imgsz=YOLO_IMGSZ,
                conf=YOLO_CONF,
                iou=YOLO_IOU,
                half=False,
                augment=False,
                verbose=False
            )
            fd.detections = results
            fd.t_yolo = (time.perf_counter() - t0) * 1000

            t1 = time.perf_counter()
            # Extract crops from the CLAHE-processed resized image
            crops = extract_crops(fd.clahe, fd.detections)
            panel_results = []

            if crops:
                batch = preprocess_batch(crops, CLASSIFIER_IMGSZ)
                logits_batch = self.classifier.run(
                    None, {self.cls_input_name: batch}
                )[0]
                for i, logits in enumerate(logits_batch):
                    probs = softmax(logits)
                    anomaly_prob = float(probs[1])
                    label, flag = classify_panel(anomaly_prob)
                    panel_results.append({
                        "panel_id": i + 1,
                        "label": label,
                        "confidence": float(probs.max()),
                        "healthy_prob": float(probs[0]),
                        "anomaly_prob": anomaly_prob,
                        "flag": flag
                    })

            fd.results = panel_results
            fd.t_classify = (time.perf_counter() - t1) * 1000
            self.q_save.put(fd)

    def run(self, image_paths: List[str]):
        self.warmup()
        
        if image_paths:
            self.check_yolo_determinism(image_paths[0])

        # Start all worker threads as daemon
        threads = [
            threading.Thread(target=self.resize_worker, daemon=True, name="resize"),
            threading.Thread(target=self.clahe_worker, daemon=True, name="clahe"),
            threading.Thread(target=self.detect_worker, daemon=True, name="detect"),
            threading.Thread(target=self.save_worker, daemon=True, name="save"),
        ]
        for t in threads:
            t.start()

        print(f"\nProcessing {len(image_paths)} images...")
        print("=" * 70)
        t_start = time.perf_counter()

        fed = 0
        for idx, path in enumerate(image_paths):
            t_load_start = time.perf_counter()
            img = load_image(path)
            t_load = (time.perf_counter() - t_load_start) * 1000
            
            if img is None:
                print(f"⚠️  Could not read {path}")
                continue
            
            fd = FrameData(frame_idx=idx, path=path, raw=img)
            fd.t_load = t_load
            self.q_raw.put(fd)
            fed += 1

        collected = 0
        fps_log = []

        while collected < fed:
            try:
                fd: FrameData = self.q_save.get(timeout=10)
            except queue.Empty:
                print("⚠️  Pipeline stalled")
                break

            total_ms = fd.t_load + fd.t_resize + fd.t_clahe + fd.t_yolo + fd.t_classify
            fps = 1000 / total_ms if total_ms > 0 else 0
            fps_log.append(fps)
            collected += 1

            fname = Path(fd.path).name
            anomalies = [r for r in fd.results if r["label"] == "Anomaly"]
            uncertain = [r for r in fd.results if r["label"] == "Uncertain"]

            status = "✅ All healthy"
            if anomalies:
                status = f"⚠️  {len(anomalies)} ANOMALY"
            elif uncertain:
                status = f"🟡 {len(uncertain)} UNCERTAIN"

            print(f"[{collected:03d}/{fed}] {fname:<30} | {len(fd.results)} panels | {status} | {fps:.1f} FPS")
            print(f"        Load:{fd.t_load:5.1f}ms  Resize:{fd.t_resize:5.1f}ms  CLAHE:{fd.t_clahe:5.1f}ms  YOLO:{fd.t_yolo:5.1f}ms  Classify:{fd.t_classify:5.1f}ms")

            for r in fd.results:
                print(f"        {r['flag']} Panel {r['panel_id']}: {r['label']} ({r['confidence']*100:.1f}%)")

            print()
            # No need to put back to queue since detect_worker already put to q_save
            # The save_worker processes automatically
        
        # Signal stop to all workers
        self._stop.set()
        
        # Wait a bit for save queue to empty
        print("Waiting for saves to complete...")
        timeout = 10
        start_wait = time.time()
        while not self.q_save.empty() and (time.time() - start_wait) < timeout:
            time.sleep(0.1)
        
        # Process any remaining saves synchronously
        remaining = 0
        while not self.q_save.empty():
            try:
                item = self.q_save.get_nowait()
                if item is not None:
                    self._save_result(item)
                    remaining += 1
            except queue.Empty:
                break
        
        if remaining > 0:
            print(f"Processed {remaining} remaining saves synchronously")
        
        elapsed = time.perf_counter() - t_start
        avg_fps = sum(fps_log) / len(fps_log) if fps_log else 0
        throughput = collected / elapsed

        print("=" * 70)
        print("PIPELINE SUMMARY")
        print(f"  Images processed : {collected}")
        print(f"  Wall time        : {elapsed:.2f}s")
        print(f"  Avg per-frame FPS: {avg_fps:.1f}")
        print(f"  Throughput       : {throughput:.1f} img/s")
        print(f"  Results saved to : {OUTPUT_FOLDER}")
        print("=" * 70)
        print("\n✅ Pipeline finished successfully!")

    def _save_result(self, fd: FrameData):
        # Save the CLAHE-processed resized image (what YOLO saw)
        img = fd.clahe.copy()
        if fd.detections:
            for r in fd.detections:
                if r.obb is None:
                    continue
                boxes = r.obb.xyxyxyxy.cpu().numpy()
                for i, box in enumerate(boxes):
                    pts = box.reshape(4, 2).astype(np.int32)
                    color = (0, 255, 0)
                    label_text = f"Panel {i+1}"
                    if i < len(fd.results):
                        res = fd.results[i]
                        color = (0, 0, 255) if res["label"] == "Anomaly" else (0, 255, 0)
                        if res["label"] == "Uncertain":
                            color = (0, 165, 255)
                        label_text = f"{res['label']} {res['confidence']*100:.0f}%"
                    cv2.polylines(img, [pts], isClosed=True, color=color, thickness=2)
                    cx, cy = pts.mean(axis=0).astype(int)
                    cv2.putText(img, label_text, (cx - 35, cy),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        cv2.imwrite(os.path.join(OUTPUT_FOLDER, Path(fd.path).name), img)


if __name__ == "__main__":
    exts = ["*.jpg", "*.jpeg", "*.png"]
    image_paths = []
    for ext in exts:
        image_paths.extend(sorted(glob.glob(os.path.join(IMAGE_FOLDER, ext))))

    if not image_paths:
        print(f"❌ No images found in {IMAGE_FOLDER}")
        exit(1)

    print(f"Found {len(image_paths)} images")
    try:
        Pipeline().run(image_paths)
    except KeyboardInterrupt:
        print("\n⚠️ Interrupted by user")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
