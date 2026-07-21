"""
pv_inspection_pipeline.py  —  Full PV Inspection Pipeline
==========================================================
- Per-panel independent processing (Binary + Multiclass on each YOLO crop)
- Up to 2 panels per image
- Thermal near-tie detection
- YOLO min confidence filter
"""

from __future__ import annotations

import glob
import os
from dataclasses import dataclass
from typing import List, Tuple

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
from ultralytics import YOLO

try:
    import tensorflow as tf
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False
    raise RuntimeError("TensorFlow is required for this pipeline.")


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
RGB_CLASSES = ["clean", "soiling_pollution", "shadowing_vegetation",
               "burn_discoloration", "structural_damage"]
THERMAL_MULTI_CLASSES = ["hotspot", "partial_cold", "full_cold"] 
RGB_MULTI_CLASSES = ["soiling_pollution", "shadowing_vegetation", "burn_discoloration", "structural_damage"]
RGB_BINARY_CLASSES = ["clean", "anomaly"]

THERMAL_BINARY_THRESHOLD = 0.5
CONF_THRESHOLD = 0.35
THERMAL_TIE_THRESHOLD = 0.08
YOLO_MIN_CONF = 0.25

_EFFICIENTNET_MEAN_BGR = [103.939, 116.779, 123.68]

FUSION_RULES: dict[tuple[str, str], str] = {
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

MAINTENANCE_ACTIONS: dict[str, str] = {
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

SEVERITY: dict[str, int] = {
    "healthy": 0, "soiling": 1, "shadowing": 1, "partial_cold": 2,
    "hotspot": 3, "cold_module": 3, "structural_damage": 3,
    "dead_cell": 4, "uncertain": 2,
}

_IMAGENET_MEAN = [0.485, 0.456, 0.406]
_IMAGENET_STD  = [0.229, 0.224, 0.225]


# ---------------------------------------------------------------------------
# FusionResult
# ---------------------------------------------------------------------------
@dataclass
class FusionResult:
    label: str
    confidence: float
    severity: int
    action: str
    rgb_label: str = ""
    thermal_label: str = ""
    rgb_conf: float = 0.0
    thermal_conf: float = 0.0
    rgb_probs: list = None
    thermal_probs: list = None
    low_confidence: bool = False
    rule_matched: bool = False
    stage: str = "full"
    panel_id: int = 1

    def summary(self) -> str:
        flag       = " ⚠ LOW CONF"  if self.low_confidence else ""
        stage_info = " [BINARY SKIP]" if self.stage == "binary_skip" else ""
        lines = [
            f"{'─'*60}",
            f"[FUSION Panel {self.panel_id}]{flag}{stage_info}",
            f"  Final label : {self.label.upper()}",
            f"  Severity    : {self.severity}",
            f"  Confidence  : {self.confidence:.3f}",
            f"  RGB         : {self.rgb_label} ({self.rgb_conf:.3f})",
            f"  Thermal     : {self.thermal_label} ({self.thermal_conf:.3f})",
            f"  Action      : {self.action}",
        ]
        if self.rgb_probs:
            probs_str = "  ".join(f"{c}={p:.2f}" for c, p in zip(RGB_MULTI_CLASSES, self.rgb_probs))
            lines.append(f"  RGB probs   : {probs_str}")
        if self.thermal_probs:
            probs_str = "  ".join(f"{c}={p:.2f}" for c, p in zip(THERMAL_MULTI_CLASSES, self.thermal_probs))
            lines.append(f"  Therm probs : {probs_str}")
        return "\n".join(lines)


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


def efficientnet_preprocess(image_array: np.ndarray) -> np.ndarray:
    image_array = image_array[..., ::-1]
    mean = _EFFICIENTNET_MEAN_BGR
    image_array[..., 0] -= mean[0]
    image_array[..., 1] -= mean[1]
    image_array[..., 2] -= mean[2]
    return image_array


# ---------------------------------------------------------------------------
# Model loaders (unchanged)
# ---------------------------------------------------------------------------
def _load_rgb_binary(path: str) -> torch.nn.Module:
    from torchvision.models import mobilenet_v3_small
    checkpoint = torch.load(path, map_location="cpu")
    if isinstance(checkpoint, torch.nn.Module):
        print("✅ RGB binary: loaded as full Module")
        return checkpoint.eval()
    state_dict = checkpoint.get("state_dict", checkpoint)
    model = mobilenet_v3_small(weights=None)
    in_features = model.classifier[3].in_features
    model.classifier[3] = torch.nn.Linear(in_features, 2)
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing:
        print(f"  ⚠ RGB binary — missing keys: {missing[:10]}")
    if unexpected:
        print(f"  ⚠ RGB binary — unexpected keys: {unexpected[:10]}")
    print("✅ RGB binary loaded (MobileNetV3-Small, 2 classes)")
    return model.eval()


def _load_rgb_multi(path: str) -> torch.nn.Module:
    from torchvision.models import efficientnet_b1
    checkpoint = torch.load(path, map_location="cpu")
    if isinstance(checkpoint, torch.nn.Module):
        print("✅ RGB multi: loaded as full Module")
        return checkpoint.eval()
    state_dict = checkpoint.get("state_dict", checkpoint)
    num_classes = 5
    for k, v in state_dict.items():
        if "classifier.1.weight" in k:
            num_classes = v.shape[0]
            break
    model = efficientnet_b1(weights=None)
    model.classifier[1] = torch.nn.Linear(model.classifier[1].in_features, num_classes)
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing:
        print(f"  ⚠ RGB multi — missing keys  : {missing[:5]}")
    print(f"✅ RGB multi loaded (EfficientNet-B1, {num_classes} classes)")
    return model.eval()


def _load_thermal_multi(path: str):
    interp = tf.lite.Interpreter(model_path=path)
    interp.allocate_tensors()
    inp = interp.get_input_details()[0]
    print(f"✅ Thermal multi TFLite loaded — input shape: {inp['shape']}  dtype: {inp['dtype'].__name__}")
    return interp


# ---------------------------------------------------------------------------
# PVInspectionPipeline
# ---------------------------------------------------------------------------
class PVInspectionPipeline:
    YOLO_PATH          = "/Users/mac/Documents/PFE/panel_detection/yolov11-1000/best.pt"
    RGB_BINARY_PATH    = "/Users/mac/Documents/PFE/rgb_binary_class/best_model_mobileNEtEnhanced.(2).pt"
    RGB_MULTI_PATH     = "/Users/mac/Documents/PFE/rgb_multicalssification/best_rgb_multiclass_final_last.pt"
    THERMAL_BIN_PATH   = "/Users/mac/Documents/PFE/thermal_binary_class/thermal_binary_mobilenetv2.keras"
    THERMAL_MULTI_PATH = "/Users/mac/Documents/PFE/thermal_multiclassification/thermal_multiclass_efficientnetb1_f16.tflite"
    TEST_FOLDER        = "/Users/mac/Documents/PFE/fusion/testing"

    CONF_THRESHOLD = 0.35
    YOLO_MIN_CONF = 0.25
    THERMAL_TIE_THRESHOLD = 0.08

    def __init__(self):
        print("\n" + "="*60)
        print("Loading models…")
        self.yolo = YOLO(self.YOLO_PATH)
        self.rgb_binary = _load_rgb_binary(self.RGB_BINARY_PATH)
        self.rgb_multi = _load_rgb_multi(self.RGB_MULTI_PATH)
        self.thermal_binary = tf.keras.models.load_model(self.THERMAL_BIN_PATH)
        self.thermal_multi = _load_thermal_multi(self.THERMAL_MULTI_PATH)
        self._therm_multi_size = tuple(
            self.thermal_multi.get_input_details()[0]["shape"][1:3]
        )
        print("All models loaded.\n" + "="*60)

    def _read_rgb(self, path: str) -> np.ndarray:
        return cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)

    def _read_thermal(self, path: str) -> np.ndarray:
        return np.array(Image.open(path).convert("RGB"))

    def _yolo_crop(self, img_rgb: np.ndarray, rgb_path: str) -> List[np.ndarray]:
        print(f"    [YOLO DEBUG] Input shape: {img_rgb.shape}")

        results = self.yolo(img_rgb, verbose=False, conf=0.1, imgsz=1024, iou=0.4, max_det=5, augment=False)

        detections = []
        for r in results:
            if hasattr(r, 'obb') and r.obb is not None and len(r.obb.xyxy) > 0:
                for box, conf in zip(r.obb.xyxy.cpu().numpy(), r.obb.conf.cpu().numpy()):
                    if conf >= self.YOLO_MIN_CONF:
                        x1, y1, x2, y2 = map(int, box[:4])
                        h, w = img_rgb.shape[:2]
                        x1, y1, x2, y2 = max(0, x1), max(0, y1), min(w, x2), min(h, y2)
                        if x2 > x1 and y2 > y1:
                            crop = img_rgb[y1:y2, x1:x2]
                            detections.append((conf, crop, r))

        detections.sort(key=lambda x: x[0], reverse=True)

        if not detections:
            print("    (YOLO: no reliable detection — using full image)")
            vis = results[0].plot() if results else img_rgb
            self._save_detection_vis(vis, rgb_path, success=False)
            return [img_rgb]

        top_crops = [det[1] for det in detections[:2]]
        print(f"    (YOLO: using {len(top_crops)} panel(s) — best conf={detections[0][0]:.3f})")
        
        vis = detections[0][2].plot()
        self._save_detection_vis(vis, rgb_path, success=True)
        return top_crops

    def _save_detection_vis(self, vis_img: np.ndarray, rgb_path: str, success: bool = True):
        base = os.path.basename(rgb_path).replace("_rgb.png", "")
        out_path = os.path.join(os.path.dirname(rgb_path), f"{base}_yolo_detection.png")
        
        if isinstance(vis_img, np.ndarray) and len(vis_img.shape) == 3 and vis_img.shape[2] == 3:
            save_img = cv2.cvtColor(vis_img, cv2.COLOR_RGB2BGR)
        else:
            save_img = vis_img

        if not success:
            cv2.putText(save_img, "NO DETECTION", (30, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)

        cv2.imwrite(out_path, save_img)
        print(f"    → Saved: {os.path.basename(out_path)}")

    # --------------------- Per Panel Processing ---------------------
    def process_panel(self, crop_rgb: np.ndarray, img_therm: np.ndarray, panel_id: int) -> FusionResult:
        # Binary stage
        tensor = to_tensor_224(crop_rgb)
        with torch.no_grad():
            probs = F.softmax(self.rgb_binary(tensor)[0], dim=0).numpy()
        rgb_bin_label = RGB_BINARY_CLASSES[int(np.argmax(probs))]
        rgb_bin_conf = float(probs[int(np.argmax(probs))])

        therm_bin_label, therm_bin_conf = self.predict_thermal_binary(img_therm)

        print(f"    Panel {panel_id} - Binary RGB  : {rgb_bin_label} ({rgb_bin_conf:.3f})")
        print(f"    Panel {panel_id} - Binary Therm: {therm_bin_label} ({therm_bin_conf:.3f})")

        # 1. If BOTH are healthy, skip entirely
        if rgb_bin_label == "clean" and therm_bin_label == "no_anomaly":
            return FusionResult(
                label="healthy",
                confidence=min(rgb_bin_conf, therm_bin_conf),
                severity=0,
                action="No action required",
                rgb_label="clean",
                thermal_label="no_anomaly",
                rgb_conf=rgb_bin_conf,
                thermal_conf=therm_bin_conf,
                stage="binary_skip",
                panel_id=panel_id
            )

        # 2. Independent Multiclass Execution (Run ONLY on the modality flagged as anomaly)
        
        # --- RGB Modality ---
        if rgb_bin_label == "anomaly":
            rgb_probs_arr = self.predict_rgb_multi(crop_rgb)
            rgb_idx = int(np.argmax(rgb_probs_arr))
            final_rgb_label = RGB_MULTI_CLASSES[rgb_idx]
            final_rgb_conf = float(rgb_probs_arr[rgb_idx])
            rgb_probs_list = rgb_probs_arr.tolist()
            print(f"    Panel {panel_id} - RGB probs : { {c: f'{p:.3f}' for c, p in zip(RGB_MULTI_CLASSES, rgb_probs_arr)} }")
        else:
            # Retain the clean state and skip multiclass
            final_rgb_label = "clean"
            final_rgb_conf = rgb_bin_conf
            rgb_probs_list = []
            print(f"    Panel {panel_id} - RGB Multiclass SKIPPED (kept 'clean')")

        # --- Thermal Modality ---
        if therm_bin_label == "anomaly":
            therm_probs_arr = self.predict_thermal_multi(img_therm)
            therm_idx = int(np.argmax(therm_probs_arr))
            final_therm_label = THERMAL_MULTI_CLASSES[therm_idx]
            final_therm_conf = float(therm_probs_arr[therm_idx])
            therm_probs_list = therm_probs_arr.tolist()
            print(f"    Panel {panel_id} - Therm probs : { {c: f'{p:.3f}' for c, p in zip(THERMAL_MULTI_CLASSES, therm_probs_arr)} }")
        else:
            # Retain the clean state and skip multiclass
            final_therm_label = "no_anomaly"
            final_therm_conf = therm_bin_conf
            therm_probs_list = []
            print(f"    Panel {panel_id} - Therm Multiclass SKIPPED (kept 'no_anomaly')")

        # 3. Fuse the independent results
        stage_type = "hybrid" if (not rgb_probs_list or not therm_probs_list) else "full"
        return self._fuse(
            rgb_label=final_rgb_label, 
            rgb_conf=final_rgb_conf, 
            rgb_probs=rgb_probs_list,
            thermal_label=final_therm_label, 
            thermal_conf=final_therm_conf, 
            thermal_probs=therm_probs_list,
            stage=stage_type, 
            panel_id=panel_id
        )

    def predict_thermal_binary(self, img_therm: np.ndarray) -> Tuple[str, float]:
        clahe = apply_clahe(img_therm)
        arr = np.expand_dims(
            np.array(Image.fromarray(clahe).resize((224, 224)), dtype=np.float32) / 255.0, axis=0
        )
        raw = self.thermal_binary.predict(arr, verbose=0)[0]
        if raw.shape[0] == 1:
            prob_no = float(raw[0])
            return ("no_anomaly", prob_no) if prob_no > THERMAL_BINARY_THRESHOLD else ("anomaly", 1.0 - prob_no)
        else:
            idx = int(np.argmax(raw))
            label = "anomaly" if idx == 1 else "no_anomaly"
            return label, float(raw[idx])

    def predict_rgb_multi(self, img_rgb: np.ndarray) -> np.ndarray:
        tensor = to_tensor_224(img_rgb)
        with torch.no_grad():
            probs = F.softmax(self.rgb_multi(tensor)[0], dim=0).numpy()
        return probs

    def predict_thermal_multi(self, img_therm: np.ndarray) -> np.ndarray:
        H, W = self._therm_multi_size
        img = Image.fromarray(img_therm).resize((W, H), Image.Resampling.BILINEAR)
        img_array = np.array(img, dtype=np.float32)
        img_array = efficientnet_preprocess(img_array)
        img_array = np.expand_dims(img_array, axis=0)

        interp = self.thermal_multi
        inp = interp.get_input_details()[0]
        out = interp.get_output_details()[0]

        input_data = img_array.astype(np.uint8 if inp["dtype"] == np.uint8 else np.float32)
        interp.set_tensor(inp["index"], input_data)
        interp.invoke()
        probs = interp.get_tensor(out["index"])[0].astype(np.float32)
        print(f"    [DEBUG] Thermal multi raw probs: {probs}")
        return probs

    def _fuse(self, rgb_label: str, rgb_conf: float, rgb_probs: list, 
              thermal_label: str, thermal_conf: float, thermal_probs: list, 
              stage: str = "full", panel_id: int = 1) -> FusionResult:
        
        confidence = min(rgb_conf, thermal_conf)
        low_conf = confidence < self.CONF_THRESHOLD

        # Only check thermal tie-threshold if the thermal multiclassifier was actually run
        if thermal_probs and len(thermal_probs) >= 2:
            sorted_therm = np.sort(thermal_probs)[::-1]
            if sorted_therm[0] - sorted_therm[1] < self.THERMAL_TIE_THRESHOLD:
                low_conf = True
                print(f"    [INFO] Thermal near-tie detected → uncertain")

        if low_conf:
            label = "uncertain"
            rule_matched = False
        else:
            # Map directly to FUSION_RULES using whatever labels were resolved above
            key = (rgb_label, thermal_label)
            label = FUSION_RULES.get(key, "uncertain")
            rule_matched = True

        return FusionResult(
            label=label, confidence=confidence, severity=SEVERITY.get(label, 2),
            action=MAINTENANCE_ACTIONS.get(label, "Manual re-inspection required"),
            rgb_label=rgb_label, thermal_label=thermal_label,
            rgb_conf=rgb_conf, thermal_conf=thermal_conf,
            rgb_probs=rgb_probs, thermal_probs=thermal_probs,
            low_confidence=low_conf, rule_matched=rule_matched, stage=stage,
            panel_id=panel_id
        )

    def process_pair(self, rgb_path: str, thermal_path: str) -> List[FusionResult]:
        img_rgb = self._read_rgb(rgb_path)
        img_therm = self._read_thermal(thermal_path)

        crops = self._yolo_crop(img_rgb, rgb_path)

        results = []
        for i, crop in enumerate(crops, 1):
            print(f"\n  --- Processing Panel {i} ---")
            result = self.process_panel(crop, img_therm, panel_id=i)
            results.append(result)

        return results

    def run_test_folder(self, folder: str | None = None) -> None:
        folder = folder or self.TEST_FOLDER
        rgb_files = sorted(glob.glob(os.path.join(folder, "*_rgb.png")))
        if not rgb_files:
            print(f"No *_rgb.png files found in {folder}")
            return

        print(f"\nFound {len(rgb_files)} RGB images in {folder}\n")
        for rgb_path in rgb_files:
            base = os.path.basename(rgb_path).replace("_rgb.png", "")
            thermal_path = os.path.join(folder, f"{base}_thermal.png")
            if not os.path.exists(thermal_path):
                print(f"⚠ Missing thermal for {base} — skipped")
                continue

            print(f"\n🔍  {base}")
            results = self.process_pair(rgb_path, thermal_path)
            for result in results:
                print(result.summary())

        print("\n" + "="*60 + "\nDone.\n")


# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("\n" + "="*60)
    print("  PV INSPECTION PIPELINE — FULL TEST")
    print("="*60)
    pipeline = PVInspectionPipeline()
    pipeline.run_test_folder()