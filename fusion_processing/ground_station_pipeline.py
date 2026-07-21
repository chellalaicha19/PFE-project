"""
ground_station_pipeline.py — Ground Station Multiclass & Fusion
==============================================================
- Runs on Ground PC
- Processes only the "anomalous" pairs forwarded by the Pi 5
- Runs Deep Multiclass Models (RGB + Thermal)
- Executes final lookup rules, structural assessment, and actionable maintenance reporting
"""

from __future__ import annotations
import os
import glob
from dataclasses import dataclass
import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
from ultralytics import YOLO
import tensorflow as tf

# ---------------------------------------------------------------------------
# Constants & Rules Mapping
# ---------------------------------------------------------------------------
RGB_MULTI_CLASSES = ["soiling_pollution", "shadowing_vegetation", "burn_discoloration", "structural_damage"]
THERMAL_MULTI_CLASSES = ["hotspot", "partial_cold", "full_cold"]

CONF_THRESHOLD = 0.35
THERMAL_TIE_THRESHOLD = 0.08
YOLO_MIN_CONF = 0.25

FUSION_RULES = {
    ("clean",               "no_anomaly"  ): "healthy",
    ("clean",               "hotspot"     ): "hotspot",
    ("clean",               "partial_cold"): "partial_cold",
    ("clean",               "full_cold"   ): "cold_module",
    ("soiling_pollution",   "no_anomaly"  ): "soiling",
    ("soiling_pollution",   "hotspot"     ): "hotspot",
    ("soiling_pollution",   "partial_cold"): "partial_cold",
    ("soiling_pollution",   "full_cold"   ): "cold_module",
    ("shadowing_vegetation","no_anomaly"  ): "shadowing",
    ("shadowing_vegetation","hotspot"     ): "hotspot",
    ("shadowing_vegetation","partial_cold"): "partial_cold",
    ("shadowing_vegetation","full_cold"   ): "cold_module",
    ("burn_discoloration",  "no_anomaly"  ): "hotspot",
    ("burn_discoloration",  "hotspot"     ): "hotspot",
    ("burn_discoloration",  "partial_cold"): "hotspot",
    ("burn_discoloration",  "full_cold"   ): "dead_cell",
    ("structural_damage",   "no_anomaly"  ): "structural_damage",
    ("structural_damage",   "hotspot"     ): "structural_damage",
    ("structural_damage",   "partial_cold"): "structural_damage",
    ("structural_damage",   "full_cold"   ): "dead_cell",
}

MAINTENANCE_ACTIONS = {
    "healthy":           "No action required",
    "soiling":           "Schedule panel cleaning",
    "shadowing":         "Trim vegetation or reposition panels",
    "partial_cold":      "Inspect bypass diodes and string connections",
    "hotspot":           "Urgent inspection — possible cell replacement",
    "cold_module":       "Check string connection and inverter input",
    "dead_cell":         "Module replacement required",
    "structural_damage": "Physical inspection and replacement",
    "uncertain":         "Manual re-inspection required",
}

SEVERITY = {
    "healthy": 0, "soiling": 1, "shadowing": 1, "partial_cold": 2,
    "hotspot": 3, "cold_module": 3, "structural_damage": 3,
    "dead_cell": 4, "uncertain": 2,
}

_IMAGENET_MEAN = [0.485, 0.456, 0.406]
_IMAGENET_STD  = [0.229, 0.224, 0.225]
_EFFICIENTNET_MEAN_BGR = [103.939, 116.779, 123.68]

# Update these paths for your Ground PC setup
YOLO_PATH          = "/Users/mac/Documents/PFE/panel_detection/yolov11-1000/best.pt"
RGB_MULTI_PATH     = "/Users/mac/Documents/PFE/rgb_multicalssification/best_rgb_multiclass_final_last.pt"
THERMAL_MULTI_PATH = "/Users/mac/Documents/PFE/thermal_multiclassification/thermal_multiclass_efficientnetb1_f16.tflite"
ANOMALOUS_DIR      = "/Users/mac/Documents/PFE/fusion/testing"

# ---------------------------------------------------------------------------
# Data Containers & Data Handling Helpers
# ---------------------------------------------------------------------------
@dataclass
class FusionResult:
    label: str; confidence: float; severity: int; action: str
    rgb_label: str = ""; thermal_label: str = ""
    rgb_conf: float = 0.0; thermal_conf: float = 0.0
    rgb_probs: list = None; thermal_probs: list = None
    low_confidence: bool = False; panel_id: int = 1

    def summary(self) -> str:
        flag = " ⚠ LOW CONF" if self.low_confidence else ""
        lines = [
            f"{'─'*60}",
            f"[FUSION Ground Station Analysis — Panel {self.panel_id}]{flag}",
            f"  Diagnosis   : {self.label.upper()}",
            f"  Severity    : Level {self.severity}",
            f"  Confidence  : {self.confidence:.3f}",
            f"  RGB Multi   : {self.rgb_label} ({self.rgb_conf:.3f})",
            f"  Thermal Multi: {self.thermal_label} ({self.thermal_conf:.3f})",
            f"  Action Item : {self.action}",
        ]
        return "\n".join(lines)

def apply_clahe(image_rgb: np.ndarray) -> np.ndarray:
    lab = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    return cv2.cvtColor(cv2.merge((clahe.apply(l), a, b)), cv2.COLOR_LAB2RGB)

def to_tensor_224(image_rgb: np.ndarray) -> torch.Tensor:
    pil = Image.fromarray(apply_clahe(image_rgb))
    tf_ = transforms.Compose([
        transforms.Resize((224, 224)), transforms.ToTensor(),
        transforms.Normalize(_IMAGENET_MEAN, _IMAGENET_STD),
    ])
    return tf_(pil).unsqueeze(0)

def efficientnet_preprocess(image_array: np.ndarray) -> np.ndarray:
    image_array = image_array[..., ::-1]
    image_array[..., 0] -= _EFFICIENTNET_MEAN_BGR[0]
    image_array[..., 1] -= _EFFICIENTNET_MEAN_BGR[1]
    image_array[..., 2] -= _EFFICIENTNET_MEAN_BGR[2]
    return image_array

def load_rgb_multi(path: str) -> torch.nn.Module:
    from torchvision.models import efficientnet_b1
    checkpoint = torch.load(path, map_location="cpu")
    if isinstance(checkpoint, torch.nn.Module): return checkpoint.eval()
    state_dict = checkpoint.get("state_dict", checkpoint)
    num_classes = 5
    for k, v in state_dict.items():
        if "classifier.1.weight" in k:
            num_classes = v.shape[0]
            break
    model = efficientnet_b1(weights=None)
    model.classifier[1] = torch.nn.Linear(model.classifier[1].in_features, num_classes)
    model.load_state_dict(state_dict, strict=False)
    return model.eval()

# ---------------------------------------------------------------------------
# Ground Station Class
# ---------------------------------------------------------------------------
class GroundStationPipeline:
    def __init__(self):
        print("🖥️ Initializing Heavy Diagnostics Pipeline on Ground Station...")
        self.yolo = YOLO(YOLO_PATH)
        self.rgb_multi = load_rgb_multi(RGB_MULTI_PATH)
        
        self.thermal_multi = tf.lite.Interpreter(model_path=THERMAL_MULTI_PATH)
        self.thermal_multi.allocate_tensors()
        self._therm_multi_size = tuple(self.thermal_multi.get_input_details()[0]["shape"][1:3])
        print("✅ Core multiclass engines fully initialized.\n")

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

    def predict_rgb_multi(self, img_rgb: np.ndarray) -> np.ndarray:
        tensor = to_tensor_224(img_rgb)
        with torch.no_grad():
            probs = F.softmax(self.rgb_multi(tensor)[0], dim=0).numpy()
        return probs

    def predict_thermal_multi(self, img_therm: np.ndarray) -> np.ndarray:
        H, W = self._therm_multi_size
        img = Image.fromarray(img_therm).resize((W, H), Image.Resampling.BILINEAR)
        img_array = efficientnet_preprocess(np.array(img, dtype=np.float32))
        img_array = np.expand_dims(img_array, axis=0)

        inp = self.thermal_multi.get_input_details()[0]
        out = self.thermal_multi.get_output_details()[0]
        
        input_data = img_array.astype(np.uint8 if inp["dtype"] == np.uint8 else np.float32)
        self.thermal_multi.set_tensor(inp["index"], input_data)
        self.thermal_multi.invoke()
        return self.thermal_multi.get_tensor(out["index"])[0].astype(np.float32)

    def _fuse(self, rgb_probs: np.ndarray, thermal_probs: np.ndarray, panel_id: int) -> FusionResult:
        rgb_idx = int(np.argmax(rgb_probs))
        thermal_idx = int(np.argmax(thermal_probs))
        
        # Guard mapping logic against unexpected output shapes
        rgb_label = RGB_MULTI_CLASSES[rgb_idx] if rgb_idx < len(RGB_MULTI_CLASSES) else "clean"
        thermal_label = THERMAL_MULTI_CLASSES[thermal_idx]

        confidence = min(float(rgb_probs[rgb_idx]), float(thermal_probs[thermal_idx]))
        low_conf = confidence < CONF_THRESHOLD

        if len(thermal_probs) >= 2:
            sorted_therm = np.sort(thermal_probs)[::-1]
            if sorted_therm[0] - sorted_therm[1] < THERMAL_TIE_THRESHOLD:
                low_conf = True

        if low_conf:
            label = "uncertain"
        else:
            label = FUSION_RULES.get((rgb_label, thermal_label), "uncertain")

        return FusionResult(
            label=label, confidence=confidence, severity=SEVERITY.get(label, 2),
            action=MAINTENANCE_ACTIONS.get(label, "Manual re-inspection required"),
            rgb_label=rgb_label, thermal_label=thermal_label,
            rgb_conf=float(rgb_probs[rgb_idx]), thermal_conf=float(thermal_probs[thermal_idx]),
            rgb_probs=rgb_probs.tolist(), thermal_probs=thermal_probs.tolist(),
            low_confidence=low_conf, panel_id=panel_id
        )

    def process_anomalous_folder(self):
        rgb_files = sorted(glob.glob(os.path.join(ANOMALOUS_DIR, "*_rgb.png")))
        if not rgb_files:
            print(f"No anomalous images found in path: {ANOMALOUS_DIR}")
            return

        print(f"Found {len(rgb_files)} flagged entries requiring advanced diagnostics.\n")
        for rgb_path in rgb_files:
            base = os.path.basename(rgb_path).replace("_rgb.png", "")
            thermal_path = os.path.join(ANOMALOUS_DIR, f"{base}_thermal.png")
            
            print(f"\n🔍 Analyzing Complex Defect Match: {base}")
            img_rgb = cv2.cvtColor(cv2.imread(rgb_path), cv2.COLOR_BGR2RGB)
            img_therm = np.array(Image.open(thermal_path).convert("RGB"))
            
            crops = self.yolo_crop(img_rgb)
            for i, crop in enumerate(crops, 1):
                rgb_probs = self.predict_rgb_multi(crop)
                therm_probs = self.predict_thermal_multi(img_therm)
                result = self._fuse(rgb_probs, therm_probs, panel_id=i)
                print(result.summary())

if __name__ == "__main__":
    ground_station = GroundStationPipeline()
    ground_station.process_anomalous_folder()