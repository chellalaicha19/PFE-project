"""
core_inference.py — Ground-station multiclass + fusion stage
==============================================================
Takes anomaly frame folders produced on the Pi 5 (binary stage already run
onboard) and runs:
  1. YOLO panel detection on rgb.jpg -> crop
  2. RGB EfficientNet-B1 multiclass on the crop
  3. Thermal EfficientNet-B1 (TFLite) multiclass on the full thermal.jpg
  4. FusionEngine rule lookup -> final diagnostic label

Extracted from the full onboard+ground pipeline (fusion2.py), keeping only
the ground-station multiclass + fusion stage — the binary stage is assumed
already done on the Pi 5.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from typing import List, Optional

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
from ultralytics import YOLO

import tensorflow as tf


# ---------------------------------------------------------------------------
# Config — adjust paths here if models move
# ---------------------------------------------------------------------------
YOLO_PATH          = "/Users/mac/Documents/PFE/panel_detection/yolov11-1000/best.pt"
RGB_MULTI_PATH      = "/Users/mac/Documents/PFE/rgb_multicalssification/best_rgb_multiclass_final_last.pt"
THERMAL_MULTI_PATH  = "/Users/mac/Documents/PFE/thermal_multiclassification/thermal_multiclass_efficientnetb1_f16.tflite"

RGB_MULTI_CLASSES = ["soiling_pollution", "shadowing_vegetation", "burn_discoloration", "structural_damage"]
THERMAL_MULTI_CLASSES = ["hotspot", "partial_cold", "full_cold"]

CONF_THRESHOLD = 0.35
THERMAL_TIE_THRESHOLD = 0.08
YOLO_MIN_CONF = 0.25

_IMAGENET_MEAN = [0.485, 0.456, 0.406]
_IMAGENET_STD = [0.229, 0.224, 0.225]
_EFFICIENTNET_MEAN_BGR = [103.939, 116.779, 123.68]

# Fusion table is defined over (rgb_label, thermal_label). Since the binary
# stage already ran on the Pi and only anomalies reach this stage, rgb_label
# and thermal_label here are always drawn from the multiclass sets above —
# the "clean" / "no_anomaly" rows are kept only for completeness / reuse.
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


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------
@dataclass
class PanelResult:
    panel_id: int
    label: str
    confidence: float
    severity: int
    action: str
    rgb_label: str
    thermal_label: str
    rgb_conf: float
    thermal_conf: float
    rgb_probs: List[float] = field(default_factory=list)
    thermal_probs: List[float] = field(default_factory=list)
    low_confidence: bool = False
    rgb_stage: str = "multiclass"      # "multiclass" or "binary_skip"
    thermal_stage: str = "multiclass"  # "multiclass" or "binary_skip"

    def to_dict(self) -> dict:
        return {
            "panel_id": self.panel_id,
            "label": self.label,
            "confidence": round(self.confidence, 4),
            "severity": self.severity,
            "action": self.action,
            "rgb_label": self.rgb_label,
            "rgb_conf": round(self.rgb_conf, 4),
            "rgb_probs": {c: round(float(p), 4) for c, p in zip(RGB_MULTI_CLASSES, self.rgb_probs)},
            "rgb_stage": self.rgb_stage,
            "thermal_label": self.thermal_label,
            "thermal_conf": round(self.thermal_conf, 4),
            "thermal_probs": {c: round(float(p), 4) for c, p in zip(THERMAL_MULTI_CLASSES, self.thermal_probs)},
            "thermal_stage": self.thermal_stage,
            "low_confidence": self.low_confidence,
        }


# ---------------------------------------------------------------------------
# metadata.txt parsing (binary-stage results already produced on the Pi)
# ---------------------------------------------------------------------------
def parse_metadata(text: str) -> dict:
    """
    Reads the Pi's per-frame metadata.txt to recover the binary-stage verdicts,
    so the ground station only multiclasses whatever was actually flagged.

    Expects lines like:
        Thermal Label: NO_ANOMALY
        Thermal Confidence: 0.5301
        Panel 1: Healthy (prob: 0.0310, conf: 0.9690)
        Panel 2: Anomaly (prob: 0.9884, conf: 0.9884)
    """
    result = {"thermal_label": None, "thermal_conf": None, "panels": {}, "gps": None}

    # New format: separate "Latitude (filtered):" / "Longitude (filtered):" lines,
    # plus altitude, satellites, HDOP, and fix type.
    lat_m = re.search(r"Latitude \(filtered\):\s*([-\d.]+)", text, re.IGNORECASE)
    lon_m = re.search(r"Longitude \(filtered\):\s*([-\d.]+)", text, re.IGNORECASE)
    if lat_m and lon_m:
        gps = {"lat": float(lat_m.group(1)), "lon": float(lon_m.group(1))}

        m = re.search(r"Altitude \(filtered\):\s*([-\d.]+)", text, re.IGNORECASE)
        if m:
            gps["alt"] = float(m.group(1))
        m = re.search(r"Satellites:\s*(\d+)", text, re.IGNORECASE)
        if m:
            gps["sats"] = int(m.group(1))
        m = re.search(r"HDOP:\s*([\d.]+)", text, re.IGNORECASE)
        if m:
            gps["hdop"] = float(m.group(1))
        m = re.search(r"Fix Type:\s*\d+\s*\(([^)]+)\)", text, re.IGNORECASE)
        if m:
            gps["fix_type"] = m.group(1).strip()
        m = re.search(r"Fix Count:\s*(\d+)", text, re.IGNORECASE)
        if m:
            gps["fix_count"] = int(m.group(1))

        result["gps"] = gps
    else:
        # Old format fallback: "GPS Fix #N: lat, lon | sats=X, hdop=Y"
        m = re.search(
            r"GPS Fix #\d+:\s*([-\d.]+),\s*([-\d.]+)\s*\|\s*sats=(\d+),\s*hdop=([\d.]+)",
            text, re.IGNORECASE,
        )
        if m:
            result["gps"] = {
                "lat": float(m.group(1)),
                "lon": float(m.group(2)),
                "sats": int(m.group(3)),
                "hdop": float(m.group(4)),
            }

    m = re.search(r"Thermal Label:\s*(\S+)", text, re.IGNORECASE)
    if m:
        result["thermal_label"] = m.group(1).strip().upper()

    m = re.search(r"Thermal Confidence:\s*([\d.]+)", text, re.IGNORECASE)
    if m:
        result["thermal_conf"] = float(m.group(1))

    for match in re.finditer(
        r"Panel\s+(\d+):\s*(Healthy|Anomaly)\s*\(prob:\s*([\d.]+),\s*conf:\s*([\d.]+)\)",
        text, re.IGNORECASE,
    ):
        panel_id = int(match.group(1))
        result["panels"][panel_id] = {
            "label": match.group(2).strip().lower(),  # "healthy" or "anomaly"
            "prob": float(match.group(3)),
            "conf": float(match.group(4)),
        }

    return result


# ---------------------------------------------------------------------------
# Preprocessing helpers (unchanged from fusion2.py)
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
    image_array = image_array[..., ::-1].astype(np.float32)
    mean = _EFFICIENTNET_MEAN_BGR
    image_array[..., 0] -= mean[0]
    image_array[..., 1] -= mean[1]
    image_array[..., 2] -= mean[2]
    return image_array


# ---------------------------------------------------------------------------
# Model loader for RGB multiclass (unchanged logic from fusion2.py)
# ---------------------------------------------------------------------------
def _load_rgb_multi(path: str) -> torch.nn.Module:
    from torchvision.models import efficientnet_b1
    checkpoint = torch.load(path, map_location="cpu")
    if isinstance(checkpoint, torch.nn.Module):
        return checkpoint.eval()
    state_dict = checkpoint.get("state_dict", checkpoint)
    num_classes = 4
    for k, v in state_dict.items():
        if "classifier.1.weight" in k:
            num_classes = v.shape[0]
            break
    model = efficientnet_b1(weights=None)
    model.classifier[1] = torch.nn.Linear(model.classifier[1].in_features, num_classes)
    model.load_state_dict(state_dict, strict=False)
    return model.eval()


def _load_thermal_multi(path: str):
    interp = tf.lite.Interpreter(model_path=path)
    interp.allocate_tensors()
    return interp


# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------
class MulticlassFusionEngine:
    """Loads models once; call process_frame() per anomaly frame folder."""

    def __init__(self,
                 yolo_path: str = YOLO_PATH,
                 rgb_multi_path: str = RGB_MULTI_PATH,
                 thermal_multi_path: str = THERMAL_MULTI_PATH):
        print("Loading models…")
        self.yolo = YOLO(yolo_path)
        self.rgb_multi = _load_rgb_multi(rgb_multi_path)
        self.thermal_multi = _load_thermal_multi(thermal_multi_path)
        self._therm_multi_size = tuple(
            self.thermal_multi.get_input_details()[0]["shape"][1:3]
        )
        print("Models loaded.")

    # ---- I/O ----
    def _read_rgb(self, path: str) -> np.ndarray:
        return cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)

    def _read_thermal(self, path: str) -> np.ndarray:
        return np.array(Image.open(path).convert("RGB"))

    # ---- YOLO panel crop (RGB only — thermal is processed whole) ----
    def _yolo_crop(self, img_rgb: np.ndarray, max_panels: int = 2) -> List[np.ndarray]:
        results = self.yolo(img_rgb, verbose=False, conf=0.1, imgsz=1024, iou=0.4, max_det=5, augment=False)

        detections = []
        for r in results:
            if hasattr(r, "obb") and r.obb is not None and len(r.obb.xyxy) > 0:
                for box, conf in zip(r.obb.xyxy.cpu().numpy(), r.obb.conf.cpu().numpy()):
                    if conf >= YOLO_MIN_CONF:
                        x1, y1, x2, y2 = map(int, box[:4])
                        h, w = img_rgb.shape[:2]
                        x1, y1, x2, y2 = max(0, x1), max(0, y1), min(w, x2), min(h, y2)
                        if x2 > x1 and y2 > y1:
                            detections.append((conf, img_rgb[y1:y2, x1:x2]))

        if not detections:
            return [img_rgb]  # fall back to full frame

        detections.sort(key=lambda x: x[0], reverse=True)
        return [d[1] for d in detections[:max_panels]]

    # ---- multiclass predictions ----
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
        return interp.get_tensor(out["index"])[0].astype(np.float32)

    # ---- fusion ----
    def _fuse(self, rgb_label, rgb_conf, rgb_probs, thermal_label, thermal_conf, thermal_probs,
              panel_id, rgb_stage="multiclass", thermal_stage="multiclass") -> PanelResult:
        confidence = min(rgb_conf, thermal_conf)
        low_conf = confidence < CONF_THRESHOLD

        if thermal_probs is not None and len(thermal_probs) >= 2:
            sorted_therm = np.sort(thermal_probs)[::-1]
            if sorted_therm[0] - sorted_therm[1] < THERMAL_TIE_THRESHOLD:
                low_conf = True

        if low_conf:
            label = "uncertain"
        else:
            label = FUSION_RULES.get((rgb_label, thermal_label), "uncertain")

        return PanelResult(
            panel_id=panel_id, label=label, confidence=float(confidence),
            severity=SEVERITY.get(label, 2), action=MAINTENANCE_ACTIONS.get(label, "Manual re-inspection required"),
            rgb_label=rgb_label, thermal_label=thermal_label,
            rgb_conf=float(rgb_conf), thermal_conf=float(thermal_conf),
            rgb_probs=list(rgb_probs), thermal_probs=list(thermal_probs),
            low_confidence=low_conf, rgb_stage=rgb_stage, thermal_stage=thermal_stage,
        )

    # ---- public entry point ----
    def process_frame(self, frame_dir: str) -> dict:
        """frame_dir must contain rgb.jpg and thermal.jpg."""
        rgb_path = os.path.join(frame_dir, "rgb.jpg")
        thermal_path = os.path.join(frame_dir, "thermal.jpg")
        annotated_path = os.path.join(frame_dir, "annotated.jpg")
        metadata_path = os.path.join(frame_dir, "metadata.txt")

        metadata_text = ""
        if os.path.exists(metadata_path):
            with open(metadata_path, "r", errors="replace") as f:
                metadata_text = f.read()
        meta = parse_metadata(metadata_text) if metadata_text else {"thermal_label": None, "thermal_conf": None, "panels": {}}

        img_rgb = self._read_rgb(rgb_path)
        img_therm = self._read_thermal(thermal_path)

        # --- Thermal: only multiclass if the Pi flagged it as anomaly ---
        # (thermal binary is a frame-level verdict — one thermal image per frame)
        if meta["thermal_label"] is not None:
            thermal_is_anomaly = meta["thermal_label"] != "NO_ANOMALY"
            if thermal_is_anomaly:
                therm_probs = self.predict_thermal_multi(img_therm)
                therm_idx = int(np.argmax(therm_probs))
                therm_label = THERMAL_MULTI_CLASSES[therm_idx]
                therm_conf = float(therm_probs[therm_idx])
                therm_stage = "multiclass"
            else:
                therm_probs = []
                therm_label = "no_anomaly"
                therm_conf = meta["thermal_conf"] if meta["thermal_conf"] is not None else 1.0
                therm_stage = "binary_skip"
        else:
            # No metadata to go on — fall back to always running multiclass.
            therm_probs = self.predict_thermal_multi(img_therm)
            therm_idx = int(np.argmax(therm_probs))
            therm_label = THERMAL_MULTI_CLASSES[therm_idx]
            therm_conf = float(therm_probs[therm_idx])
            therm_stage = "multiclass"

        # --- RGB: per-panel, only multiclass panels the Pi flagged as anomaly ---
        # Trust metadata.txt for *how many* panels exist (the Pi's onboard YOLO
        # already decided this) — we only re-run YOLO here to get crop pixels,
        # capped to that same count so we never invent extra panels.
        expected_panels = len(meta["panels"]) if meta["panels"] else 2
        crops = self._yolo_crop(img_rgb, max_panels=expected_panels)

        # The Pi's metadata.txt is the ground truth for how many panels this frame
        # actually has. If this ground-station YOLO pass (different thresholds/scale)
        # finds extra crops, drop them rather than inventing panels the Pi never saw.
        if meta["panels"]:
            expected_n = len(meta["panels"])
            crops = crops[:expected_n]

        panels = []
        for i, crop in enumerate(crops, 1):
            panel_meta = meta["panels"].get(i)

            if panel_meta is not None and panel_meta["label"] == "healthy":
                rgb_probs = []
                rgb_label = "clean"
                rgb_conf = panel_meta["conf"]
                rgb_stage = "binary_skip"
            elif panel_meta is not None and panel_meta["label"] == "anomaly":
                rgb_probs = self.predict_rgb_multi(crop)
                rgb_idx = int(np.argmax(rgb_probs))
                rgb_label = RGB_MULTI_CLASSES[rgb_idx]
                rgb_conf = float(rgb_probs[rgb_idx])
                rgb_stage = "multiclass"
            else:
                # No matching panel entry in metadata.txt (e.g. YOLO here found a
                # different panel count than the Pi did) — fall back to multiclass.
                rgb_probs = self.predict_rgb_multi(crop)
                rgb_idx = int(np.argmax(rgb_probs))
                rgb_label = RGB_MULTI_CLASSES[rgb_idx]
                rgb_conf = float(rgb_probs[rgb_idx])
                rgb_stage = "multiclass"

            result = self._fuse(
                rgb_label=rgb_label, rgb_conf=rgb_conf, rgb_probs=rgb_probs,
                thermal_label=therm_label, thermal_conf=therm_conf, thermal_probs=therm_probs,
                panel_id=i, rgb_stage=rgb_stage, thermal_stage=therm_stage,
            )
            panels.append(result.to_dict())

        return {
            "frame": os.path.basename(frame_dir.rstrip("/")),
            "rgb_path": rgb_path if os.path.exists(rgb_path) else None,
            "thermal_path": thermal_path if os.path.exists(thermal_path) else None,
            "annotated_path": annotated_path if os.path.exists(annotated_path) else None,
            "metadata": metadata_text,
            "gps": meta.get("gps"),
            "panels": panels,
        }