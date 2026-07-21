"""
edge_pi5_pipeline.py — Edge Binary Filtering Stage
==================================================
- Runs on Raspberry Pi 5
- Detects panels using YOLO
- Performs fast Binary Classification (RGB + Thermal)
- Applies OR fusion to sort image pairs into 'healthy' or 'anomalous' folders
"""

from __future__ import annotations
import os
import glob
import shutil
import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
from ultralytics import YOLO
import tensorflow as tf

# ---------------------------------------------------------------------------
# Constants & Configuration
# ---------------------------------------------------------------------------
RGB_BINARY_CLASSES = ["clean", "anomaly"]
THERMAL_BINARY_THRESHOLD = 0.5
YOLO_MIN_CONF = 0.25

_IMAGENET_MEAN = [0.485, 0.456, 0.406]
_IMAGENET_STD  = [0.229, 0.224, 0.225]

# Update these paths for your Pi 5 setup
YOLO_PATH        = "/Users/mac/Documents/PFE/panel_detection/yolov11-1000/best.pt"
RGB_BINARY_PATH  = "/Users/mac/Documents/PFE/rgb_binary_class/best_model_mobileNEtEnhanced.(2).pt"
THERMAL_BIN_PATH = "/Users/mac/Documents/PFE/thermal_binary_class/thermal_binary_mobilenetv2.keras"

INPUT_FOLDER     = "/Users/mac/Documents/PFE/test_rgb+thermal"
OUTPUT_BASE      = "/Users/mac/Documents/PFE/pi5_output"

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def apply_clahe(image_rgb: np.ndarray) -> np.ndarray:
    lab = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    lab_clahe = cv2.merge((clahe.apply(l), a, b))
    return cv2.cvtColor(lab_clahe, cv2.COLOR_LAB2RGB)

def to_tensor_224(image_rgb: np.ndarray) -> torch.Tensor:
    pil = Image.fromarray(apply_clahe(image_rgb))
    tf_ = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(_IMAGENET_MEAN, _IMAGENET_STD),
    ])
    return tf_(pil).unsqueeze(0)

def load_rgb_binary(path: str) -> torch.nn.Module:
    from torchvision.models import mobilenet_v3_small
    checkpoint = torch.load(path, map_location="cpu")
    if isinstance(checkpoint, torch.nn.Module):
        return checkpoint.eval()
    state_dict = checkpoint.get("state_dict", checkpoint)
    model = mobilenet_v3_small(weights=None)
    in_features = model.classifier[3].in_features
    model.classifier[3] = torch.nn.Linear(in_features, 2)
    model.load_state_dict(state_dict, strict=False)
    return model.eval()

# ---------------------------------------------------------------------------
# Pi5 Pipeline Class
# ---------------------------------------------------------------------------
class Pi5EdgePipeline:
    def __init__(self):
        print("⚡ Loading lightweight edge models on Pi 5...")
        self.yolo = YOLO(YOLO_PATH)
        self.rgb_binary = load_rgb_binary(RGB_BINARY_PATH)
        self.thermal_binary = tf.keras.models.load_model(THERMAL_BIN_PATH)
        
        self.healthy_dir = os.path.join(OUTPUT_BASE, "healthy")
        self.anomalous_dir = os.path.join(OUTPUT_BASE, "anomalous")
        os.makedirs(self.healthy_dir, exist_ok=True)
        os.makedirs(self.anomalous_dir, exist_ok=True)
        print("✅ Edge models loaded successfully.\n")

    def yolo_crop(self, img_rgb: np.ndarray) -> list[np.ndarray]:
        results = self.yolo(img_rgb, verbose=False, conf=0.1, imgsz=1024, iou=0.4, max_det=5)
        detections = []
        for r in results:
            if hasattr(r, 'obb') and r.obb is not None and len(r.obb.xyxy) > 0:
                for box, conf in zip(r.obb.xyxy.cpu().numpy(), r.obb.conf.cpu().numpy()):
                    if conf >= YOLO_MIN_CONF:
                        x1, y1, x2, y2 = map(int, box[:4])
                        h, w = img_rgb.shape[:2]
                        x1, y1, x2, y2 = max(0, x1), max(0, y1), min(w, x2), min(h, y2)
                        if x2 > x1 and y2 > y1:
                            detections.append((conf, img_rgb[y1:y2, x1:x2]))
        detections.sort(key=lambda x: x[0], reverse=True)
        return [det[1] for det in detections[:2]] if detections else [img_rgb]

    def predict_thermal_binary(self, img_therm: np.ndarray) -> str:
        clahe = apply_clahe(img_therm)
        arr = np.expand_dims(np.array(Image.fromarray(clahe).resize((224, 224)), dtype=np.float32) / 255.0, axis=0)
        raw = self.thermal_binary.predict(arr, verbose=0)[0]
        if raw.shape[0] == 1:
            return "no_anomaly" if raw[0] > THERMAL_BINARY_THRESHOLD else "anomaly"
        return "anomaly" if np.argmax(raw) == 1 else "no_anomaly"

    def predict_rgb_binary(self, crop_rgb: np.ndarray) -> str:
        tensor = to_tensor_224(crop_rgb)
        with torch.no_grad():
            probs = F.softmax(self.rgb_binary(tensor)[0], dim=0).numpy()
        return RGB_BINARY_CLASSES[int(np.argmax(probs))]

    def process_pair(self, rgb_path: str, thermal_path: str):
        img_rgb = cv2.cvtColor(cv2.imread(rgb_path), cv2.COLOR_BGR2RGB)
        img_therm = np.array(Image.open(thermal_path).convert("RGB"))

        crops = self.yolo_crop(img_rgb)
        
        # Base assumption: pair is healthy until proven otherwise
        has_anomaly = False 

        for crop in crops:
            rgb_res = self.predict_rgb_binary(crop)
            therm_res = self.predict_thermal_binary(img_therm)

            # OR Fusion logic for routing
            if rgb_res == "anomaly" or therm_res == "anomaly":
                has_anomaly = True
                break  # If even one panel is broken, the whole pair goes to Ground Station

        # Route files to appropriate directory
        target_dir = self.anomalous_dir if has_anomaly else self.healthy_dir
        shutil.copy(rgb_path, os.path.join(target_dir, os.path.basename(rgb_path)))
        shutil.copy(thermal_path, os.path.join(target_dir, os.path.basename(thermal_path)))
        
        status = "❌ ANOMALOUS -> Sent to Ground Station" if has_anomaly else "💚 HEALTHY -> Filtered Out"
        print(f"Processed {os.path.basename(rgb_path)}: {status}")

    def run(self):
        rgb_files = sorted(glob.glob(os.path.join(INPUT_FOLDER, "*_rgb.png")))
        print(f"Scanning directory. Found {len(rgb_files)} entries...")
        for rgb_path in rgb_files:
            base = os.path.basename(rgb_path).replace("_rgb.png", "")
            thermal_path = os.path.join(INPUT_FOLDER, f"{base}_thermal.png")
            if os.path.exists(thermal_path):
                self.process_pair(rgb_path, thermal_path)

if __name__ == "__main__":
    pipeline = Pi5EdgePipeline()
    pipeline.run()