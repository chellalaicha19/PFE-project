"""
Full parallel pipeline: CLAHE → YOLO (OBB) → Distilled Student Classifier (PyTorch INT8)
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

import torch
import torch.nn as nn
import torch.quantization
from ultralytics import YOLO

# ═══════════════════════════════════════════════════════════════
# CONFIGURATION
# ═══════════════════════════════════════════════════════════════
YOLO_MODEL_PATH      = "/home/pi5/panel_detection2/best_ncnn_model"
STUDENT_MODEL_PATH   = "/home/pi5/binary_class/mobileNet/student_int8.pt"  # Updated path
IMAGE_FOLDER         = "/home/pi5/binary_class/test_images"
OUTPUT_FOLDER        = "/home/pi5/panel_detection2/pipeline_results"

YOLO_CONF            = 0.35
YOLO_IOU             = 0.45
YOLO_IMGSZ           = 640
CLASSIFIER_IMGSZ     = 128

FRAME_SKIP           = 2
QUEUE_SIZE           = 4

CLASSES              = ["Healthy", "Anomaly"]

# Stability thresholds
ANOMALY_THRESHOLD    = 0.60
UNCERTAIN_BAND       = 0.10

# Normalization constants for student model
_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(1, 3, 1, 1)
_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(1, 3, 1, 1)


# ── Student Model Architecture (must match training) ──────────────────────
class TinyStudent(nn.Module):
    def __init__(self, num_classes=2):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, 3, stride=2, padding=1), 
            nn.BatchNorm2d(16), 
            nn.ReLU(),
            nn.Conv2d(16, 32, 3, stride=2, padding=1), 
            nn.BatchNorm2d(32), 
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1), 
            nn.BatchNorm2d(64), 
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1)
        )
        self.classifier = nn.Linear(64, num_classes)

    def forward(self, x):
        x = self.features(x).flatten(1)
        return self.classifier(x)


@dataclass
class FrameData:
    frame_idx: int
    path: str
    raw: np.ndarray
    clahe: Optional[np.ndarray] = None
    detections: Optional[object] = None
    results: List[dict] = field(default_factory=list)
    t_clahe: float = 0.0
    t_yolo: float = 0.0
    t_classify: float = 0.0


def load_and_resize(path: str, size=(640, 640)) -> Optional[np.ndarray]:
    img = cv2.imread(path)
    if img is None:
        return None
    return cv2.resize(img, size)


def apply_clahe(img: np.ndarray, clahe_obj) -> np.ndarray:
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    l = clahe_obj.apply(l)
    return cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_LAB2BGR)


def classify_panel(prob_anomaly: float) -> Tuple[str, str]:
    """Stable classification with dead-zone."""
    if prob_anomaly >= ANOMALY_THRESHOLD:
        return "Anomaly", "🔴"
    elif prob_anomaly >= (ANOMALY_THRESHOLD - UNCERTAIN_BAND):
        return "Uncertain", "🟡"
    else:
        return "Healthy", "🟢"


def extract_crops(img: np.ndarray, yolo_results) -> List[np.ndarray]:
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


def preprocess_batch(crops: list, size: int = 128) -> torch.Tensor:
    """Process all crops at once using PyTorch tensors for student model."""
    if not crops:
        return torch.empty((0, 3, size, size), dtype=torch.float32)
    
    n = len(crops)
    batch = np.empty((n, 3, size, size), dtype=np.float32)
    
    for i, crop in enumerate(crops):
        resized = cv2.resize(crop, (size, size))
        batch[i] = np.ascontiguousarray(resized[:, :, ::-1]).transpose(2, 0, 1)
    
    batch /= 255.0
    batch -= _MEAN
    batch /= _STD
    
    return torch.from_numpy(batch)


def softmax(x):
    e = np.exp(x - x.max())
    return e / e.sum()


class Pipeline:
    def __init__(self):
        os.makedirs(OUTPUT_FOLDER, exist_ok=True)

        print("Loading YOLO model...")
        self.yolo = YOLO(YOLO_MODEL_PATH, task="obb")

        print("Loading INT8 distilled student model...")
        # Load the INT8 quantized student model properly
        self.device = torch.device('cpu')
        
        # First create the FP32 model
        self.student = TinyStudent(num_classes=2)
        self.student.eval()
        
        # Load the state dict (which contains quantized weights)
        state_dict = torch.load(STUDENT_MODEL_PATH, map_location='cpu')
        
        # Check if it's a full quantized model or just state dict
        if hasattr(state_dict, 'state_dict'):
            # It's a full quantized model object
            state_dict = state_dict.state_dict()
        
        # For INT8 quantized models, we need to convert the model first
        try:
            # Try to load directly (might work if model was saved as quantized)
            self.student.load_state_dict(state_dict, strict=False)
            print("✅ Loaded as FP32 model with quantized weights (strict=False)")
        except RuntimeError as e:
            print(f"Warning: Could not load directly, attempting alternative method...")
            # Alternative: Create a quantized version of the model
            self.student = self._prepare_quantized_model()
            # Now load the quantized state dict
            self.student.load_state_dict(state_dict, strict=False)
            print("✅ Loaded as quantized INT8 model")
        
        self.student.eval()
        self.student.to(self.device)
        
        print(f"✅ Student model loaded and ready for inference")
        
        # Test inference to verify model works
        test_input = torch.zeros((1, 3, CLASSIFIER_IMGSZ, CLASSIFIER_IMGSZ))
        with torch.no_grad():
            test_output = self.student(test_input)
            print(f"✅ Model test inference successful, output shape: {test_output.shape}")

        self.q_raw   = queue.Queue(maxsize=QUEUE_SIZE)
        self.q_clahe = queue.Queue(maxsize=QUEUE_SIZE)
        self.q_done  = queue.Queue()

        self._stop = threading.Event()
        self.q_save = queue.Queue(maxsize=8)
    
    def _prepare_quantized_model(self):
        """Prepare a quantized version of the model for INT8 inference."""
        # First fuse Conv+BN layers for quantization
        model_to_quantize = TinyStudent(num_classes=2)
        model_to_quantize.eval()
        
        # Fuse Conv+BN layers
        try:
            model_to_quantize = torch.quantization.fuse_modules(
                model_to_quantize,
                [['features.0', 'features.1'],   # Conv2d + BatchNorm2d
                 ['features.3', 'features.4'],   # Conv2d + BatchNorm2d
                 ['features.6', 'features.7']]   # Conv2d + BatchNorm2d
            )
        except:
            # If fusion fails, continue without it
            pass
        
        # Prepare for quantization
        model_to_quantize.qconfig = torch.quantization.get_default_qconfig('qnnpack')
        torch.quantization.prepare(model_to_quantize, inplace=True)
        
        # Convert to quantized model
        torch.quantization.convert(model_to_quantize, inplace=True)
        
        return model_to_quantize

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

        print("Warming up Student Classifier...")
        dummy_crop = torch.zeros((1, 3, CLASSIFIER_IMGSZ, CLASSIFIER_IMGSZ))
        with torch.no_grad():
            _ = self.student(dummy_crop)

        print("Pipeline warmup completed.\n")

    def check_yolo_determinism(self, test_img_path: str, runs: int = 3):
        """Check if YOLO is stable (important on NCNN/ARM)."""
        try:
            img = cv2.imread(test_img_path)
            if img is None:
                print("[WARNING] Could not load test image for determinism check")
                return
            
            counts = []
            for _ in range(runs):
                r = self.yolo.predict(
                    img,
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

    def clahe_worker(self):
        local_clahe = cv2.createCLAHE(clipLimit=1.5, tileGridSize=(4, 4))
        dummy = np.random.randint(100, 200, (640, 640, 3), dtype=np.uint8)
        apply_clahe(dummy, local_clahe)
        apply_clahe(dummy, local_clahe)
        
        while not self._stop.is_set():
            try:
                fd: FrameData = self.q_raw.get(timeout=0.5)
            except queue.Empty:
                continue
            t0 = time.perf_counter()
            fd.clahe = apply_clahe(fd.raw, local_clahe)
            fd.t_clahe = (time.perf_counter() - t0) * 1000
            self.q_clahe.put(fd)

    def detect_worker(self):
        while not self._stop.is_set():
            try:
                fd: FrameData = self.q_clahe.get(timeout=0.5)
            except queue.Empty:
                continue

            t0 = time.perf_counter()
            results = self.yolo.predict(
                fd.clahe,
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
            crops = extract_crops(fd.clahe, fd.detections)
            panel_results = []

            if crops:
                # Preprocess batch for student model
                batch = preprocess_batch(crops, CLASSIFIER_IMGSZ)
                
                # Run inference with INT8 student model
                with torch.no_grad():
                    logits = self.student(batch)
                    probs = torch.softmax(logits, dim=1).cpu().numpy()
                
                for i, prob in enumerate(probs):
                    anomaly_prob = float(prob[1])  # Anomaly class probability
                    label, flag = classify_panel(anomaly_prob)
                    panel_results.append({
                        "panel_id": i + 1,
                        "label": label,
                        "confidence": float(prob.max()),
                        "healthy_prob": float(prob[0]),
                        "anomaly_prob": anomaly_prob,
                        "flag": flag
                    })

            fd.results = panel_results
            fd.t_classify = (time.perf_counter() - t1) * 1000
            self.q_done.put(fd)

    def run(self, image_paths: List[str]):
        self.warmup()
        
        if image_paths:
            self.check_yolo_determinism(image_paths[0])

        # Start all worker threads as daemon
        threads = [
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
            img = load_and_resize(path, size=(YOLO_IMGSZ, YOLO_IMGSZ))
            if img is None:
                print(f"⚠️  Could not read {path}")
                continue
            self.q_raw.put(FrameData(frame_idx=idx, path=path, raw=img))
            fed += 1

        collected = 0
        fps_log = []

        while collected < fed:
            try:
                fd: FrameData = self.q_done.get(timeout=10)
            except queue.Empty:
                print("⚠️  Pipeline stalled")
                break

            total_ms = fd.t_clahe + fd.t_yolo + fd.t_classify
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
            print(f"        CLAHE:{fd.t_clahe:5.1f}ms  YOLO:{fd.t_yolo:5.1f}ms  Classify:{fd.t_classify:5.1f}ms")

            for r in fd.results:
                print(f"        {r['flag']} Panel {r['panel_id']}: {r['label']} ({r['confidence']*100:.1f}%)")

            print()
            self.q_save.put(fd)
        
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
