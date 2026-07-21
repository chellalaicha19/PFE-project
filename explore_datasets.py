import os
import cv2
import numpy as np
import matplotlib
matplotlib.use('Agg')          # ← FORCE non-GUI backend (critical for macOS terminal)
import matplotlib.pyplot as plt
from pathlib import Path
from collections import defaultdict
import json
import warnings

# ── SUPPRESS WARNINGS ────────────────────────────────────────────────────────
warnings.filterwarnings("ignore", category=UserWarning)
os.environ["OPENCV_LOG_LEVEL"] = "ERROR"

# ── CONFIG ───────────────────────────────────────────────────────────────────
RGB_ROOT     = Path("rgb")
THERMAL_ROOT = Path("thermal")

CLASS_MAP = {
    0: "No Anomaly", 1: "Soiling/Pollution", 2: "Shadowing/Vegetation",
    3: "Hotspot-Surface", 4: "Hotspot-Internal", 5: "Cell/String Failure",
    6: "Structural Damage", 7: "Offline Module",
}

FOLDER_CLASS_MAP = {
    "Clean": 0, "Dusty": 1,
    "Bird-drop": 1, "Snow-Covered": 2,
    "Electrical-damage": 4, "Physical-Damage": 6,
}

# ── HELPERS ──────────────────────────────────────────────────────────────────
def count_images(folder, extensions=(".jpg", ".jpeg", ".png")):
    return [f for f in Path(folder).rglob("*") if f.suffix.lower() in extensions]

def safe_imread(path):
    try:
        img = cv2.imread(str(path))
        return img if img is not None else None
    except:
        return None

def sample_sizes(image_paths, n=20):
    sizes = set()
    for p in image_paths[:n]:
        img = safe_imread(p)
        if img is not None:
            sizes.add(img.shape[:2])
    return sizes if sizes else {"(no valid images)"}

def show_samples(image_paths, title, n=6):
    print(f"   📸 Generating {n} samples for {title} ...", end=" ")
    valid_paths = [p for p in image_paths if safe_imread(p) is not None][:n]
    if not valid_paths:
        print("(no valid images)")
        return
    fig, axes = plt.subplots(1, len(valid_paths), figsize=(3 * len(valid_paths), 3))
    if len(valid_paths) == 1:
        axes = [axes]
    for ax, p in zip(axes, valid_paths):
        img = cv2.cvtColor(safe_imread(p), cv2.COLOR_BGR2RGB)
        ax.imshow(img)
        ax.set_title(p.parent.name, fontsize=7)
        ax.axis("off")
    fig.suptitle(title, fontsize=10, fontweight="bold")
    plt.tight_layout()
    safe = title.replace(" ", "_").replace("/", "-").replace(":", "")
    plt.savefig(f"{safe}_samples.png", dpi=100, bbox_inches="tight")
    plt.close(fig)          # ← important: close figure so it doesn't hang
    print(f"✅ saved as {safe}_samples.png")

# ── EXPLORE FUNCTIONS (same logic) ───────────────────────────────────────────
def explore_folder_dataset(ds_path):
    ds_path = Path(ds_path)
    print(f"\n📁 {ds_path.name}  (folder-level labels)")
    all_images = []
    class_summary = defaultdict(int)

    for sub in sorted(ds_path.iterdir()):
        if not sub.is_dir(): continue
        imgs = count_images(sub)
        cls_id = FOLDER_CLASS_MAP.get(sub.name, "?")
        cls_name = CLASS_MAP.get(cls_id, "UNMAPPED") if cls_id != "?" else "UNMAPPED"
        print(f"   {sub.name:<22} → Class {cls_id} ({cls_name})  |  {len(imgs)} images")
        all_images.extend(imgs)
        if cls_id != "?": class_summary[cls_id] += len(imgs)

    print(f"   Total images : {len(all_images)}")
    print(f"   Sizes (first 20): {sample_sizes(all_images)}")
    show_samples(all_images, f"RGB {ds_path.name}")
    return all_images, class_summary

def explore_yolo_dataset(ds_path):
    ds_path = Path(ds_path)
    print(f"\n📁 {ds_path.name}  (YOLO format)")

    yaml_path = ds_path / "data.yaml"
    try:
        import yaml
        with open(yaml_path) as f:
            data = yaml.safe_load(f)
        yaml_classes = data.get("names", {})
        print(f"   YAML classes : {yaml_classes}")
    except:
        yaml_classes = {}

    all_images = []
    total_counts = defaultdict(int)

    for split in ["train", "valid", "test"]:
        split_dir = ds_path / split
        if not split_dir.exists(): continue
        imgs = count_images(split_dir / "images") if (split_dir / "images").exists() else count_images(split_dir)
        all_images.extend(imgs)
        print(f"   [{split:<5}]  images: {len(imgs):>4}")

    print(f"   ── Total images : {len(all_images)}")
    print(f"   ── Sizes (first 20): {sample_sizes(all_images)}")
    show_samples(all_images, f"RGB {ds_path.name}")
    return all_images, total_counts   # note: we simplified box counting for speed

def explore_thermal():
    ds_path = THERMAL_ROOT / "InfraredSolarModules"
    print(f"\n📁 InfraredSolarModules  (thermal)")

    images_dir = ds_path / "images"
    all_images = count_images(images_dir)
    print(f"   Total images : {len(all_images)}")
    print(f"   Sizes (first 20): {sample_sizes(all_images)}")
    show_samples(all_images, "Thermal InfraredSolarModules")
    return all_images, {}   # we'll improve thermal later if needed

# ── MAIN ──────────────────────────────────────────────────────────────────────
print("\n" + "="*70)
print("DATASET EXPLORATION STARTED")
print("="*70)

all_rgb = []
rgb_class_totals = defaultdict(int)

# 1. Detect_solar_dust
imgs, cls = explore_folder_dataset(RGB_ROOT / "Detect_solar_dust")
all_rgb.extend(imgs)
for k, v in cls.items(): rgb_class_totals[k] += v

# 2. Faulty_solar_panel
imgs, cls = explore_folder_dataset(RGB_ROOT / "Faulty_solar_panel")
all_rgb.extend(imgs)
for k, v in cls.items(): rgb_class_totals[k] += v

# 3. Solar_Panel_Defect_2
imgs, cls = explore_yolo_dataset(RGB_ROOT / "Solar_Panel_Defect_2.v4i.yolov8")
all_rgb.extend(imgs)
for k, v in cls.items(): rgb_class_totals[k] += v

# 4. solar_panel_fault_detection
imgs, cls = explore_yolo_dataset(RGB_ROOT / "solar_panel_fault_detection.v1i.yolov8")
all_rgb.extend(imgs)
for k, v in cls.items(): rgb_class_totals[k] += v

print("\n" + "="*70)
print("THERMAL DATASET")
print("="*70)
thermal_imgs, thermal_class_totals = explore_thermal()

# ── FINAL REPORT ─────────────────────────────────────────────────────────────
print("\n" + "="*70)
print("FINAL 8-CLASS COVERAGE REPORT")
print("="*70)
print(f"{'Class':<5} {'Name':<25} {'RGB imgs':>10} {'Thermal imgs':>13} {'Status'}")
print("-"*70)

for cls_id in range(8):
    name    = CLASS_MAP[cls_id]
    rgb_cnt = rgb_class_totals.get(cls_id, 0)
    th_cnt  = thermal_class_totals.get(cls_id, 0)
    total   = rgb_cnt + th_cnt
    status  = "✅ OK" if total > 50 else ("⚠️  LOW" if total > 0 else "❌ MISSING")
    print(f"  {cls_id:<4} {name:<25} {rgb_cnt:>10} {th_cnt:>13}   {status}")

print(f"\n  Total RGB images    : {len(all_rgb):,}")
print(f"  Total Thermal images: {len(thermal_imgs):,}")
print("\n✅ Exploration complete!")