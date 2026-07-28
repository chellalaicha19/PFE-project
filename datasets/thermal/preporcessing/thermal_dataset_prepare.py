"""
Thermal Dataset Preparation Script
====================================
Merges PVF-10 (train+test+Ori) and PVMD datasets into a unified split.

Pipeline per image:
  Raw image
    → [EDSR ×3 if from PVF-10]   # reduce domain gap with TC002C
    → CLAHE contrast enhancement
    → resize to 224×224
    → save to thermal_unified/

Output structure (ImageFolder-compatible):
  thermal_unified/
      train/  val/  test/
          no_anomaly/
          hotspot/
          partial_cold/
          full_cold/

Split: 70% train / 15% val / 15% test  (stratified per class)

Usage:
    pip install opencv-python-headless numpy scikit-learn tqdm
    python prepare_thermal_dataset.py
"""

import os, shutil, random
import cv2
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from tqdm import tqdm

# ─────────────────────────────────────────────
# CONFIGURATION  ← edit these if needed
# ─────────────────────────────────────────────
PVF10_ROOT  = Path("/Users/mac/Documents/PFE/thermal/PVF-10")
PVMD_ROOT   = Path("/Users/mac/Documents/PFE/thermal/PVMD dataset")
OUTPUT_ROOT = Path("/Users/mac/Documents/PFE/thermal/thermal_unified")

# Canonical class names → aliases (matched against folder names, case-insensitive substring)
CLASS_MAP = {
    "no_anomaly":   ["healthy panel", "10healthy panel", "normal", "healthy", "no_anomaly", "no anomaly", "clean"],
    "hotspot":      ["hot cell", "09hot cell", "hotspot", "Hotspots", "hot_spot", "hot spot", "bypass diode", "bypass_diode"],
    "partial_cold": ["shadow", "05shadow", "shading", "Shadings", "partial", "partial_cold", "partial cold"],
    "full_cold":    ["open circuit", "01substring open circuit", "full_cold", "full cold", "offline", "completely"],
}

USE_EDSR    = True      # EDSR ×3 for PVF-10; set False to use bicubic only
EDSR_SCALE  = 3
OUTPUT_SIZE = (224, 224)

TRAIN_RATIO = 0.70
VAL_RATIO   = 0.15
TEST_RATIO  = 0.15
SEED        = 42
IMG_EXTS    = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}

# ─────────────────────────────────────────────
# EDSR (via OpenCV DNN Super Resolution)
# ─────────────────────────────────────────────
_edsr_model = None

def check_edsr_availability():
    """Check and report EDSR availability before processing"""
    print("\n" + "=" * 60)
    print("EDSR AVAILABILITY CHECK")
    print("=" * 60)
    
    if not USE_EDSR:
        print("⚠️  EDSR is DISABLED in configuration (USE_EDSR=False)")
        print("   Will use bicubic upscaling instead.")
        return False
    
    print("✓ EDSR is ENABLED in configuration")
    
    # Check if opencv-contrib-python is installed
    try:
        import cv2
        if not hasattr(cv2, 'dnn_superres'):
            print("❌ ERROR: cv2.dnn_superres not available")
            print("   You have opencv-python but need opencv-contrib-python")
            print("   Solution: pip uninstall opencv-python && pip install opencv-contrib-python")
            return False
        print("✓ OpenCV DNN Super Resolution module is available")
    except Exception as e:
        print(f"❌ Error checking OpenCV: {e}")
        return False
    
    # Check if model file exists
    model_path = Path(__file__).parent / f"EDSR_x{EDSR_SCALE}.pb"
    if not model_path.exists():
        print(f"❌ ERROR: EDSR model not found at: {model_path}")
        print(f"   Download EDSR_x{EDSR_SCALE}.pb from:")
        print("   https://github.com/Saafke/EDSR_Tensorflow/tree/master/models")
        print(f"   Place it next to this script as 'EDSR_x{EDSR_SCALE}.pb'")
        return False
    print(f"✓ EDSR model found: {model_path}")
    print(f"  File size: {model_path.stat().st_size / 1024 / 1024:.2f} MB")
    
    # Try to load the model
    try:
        sr = cv2.dnn_superres.DnnSuperResImpl_create()
        sr.readModel(str(model_path))
        sr.setModel("edsr", EDSR_SCALE)
        print("✓ EDSR model loaded SUCCESSFULLY!")
        print(f"  Scale factor: {EDSR_SCALE}x")
        print("  Will use EDSR for PVF-10 images")
        return True
    except Exception as e:
        print(f"❌ ERROR: Failed to load EDSR model: {e}")
        return False

def load_edsr():
    global _edsr_model
    if _edsr_model is not None:
        return _edsr_model
    try:
        sr = cv2.dnn_superres.DnnSuperResImpl_create()
        model_path = Path(__file__).parent / f"EDSR_x{EDSR_SCALE}.pb"
        if not model_path.exists():
            print(f"\n[EDSR] Model not found at: {model_path}")
            print(f"  Download EDSR_x{EDSR_SCALE}.pb from:")
            print("  https://github.com/Saafke/EDSR_Tensorflow/tree/master/models")
            print(f"  Place it next to this script as 'EDSR_x{EDSR_SCALE}.pb'")
            print("  OR set USE_EDSR=False to use bicubic upscaling instead.\n")
            return None
        sr.readModel(str(model_path))
        sr.setModel("edsr", EDSR_SCALE)
        _edsr_model = sr
        print(f"[EDSR] Loaded EDSR_x{EDSR_SCALE}.pb via OpenCV DNN.")
        return _edsr_model
    except AttributeError:
        print("[EDSR] cv2.dnn_superres not available — install opencv-contrib-python.")
        return None

def upscale(img):
    model = load_edsr() if USE_EDSR else None
    if model is not None:
        try:
            return model.upsample(img)
        except Exception:
            pass
    h, w = img.shape[:2]
    return cv2.resize(img, (w * EDSR_SCALE, h * EDSR_SCALE), interpolation=cv2.INTER_CUBIC)

# ─────────────────────────────────────────────
# CLAHE
# ─────────────────────────────────────────────
def apply_clahe(img):
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    if len(img.shape) == 2:
        return clahe.apply(img)
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    return cv2.cvtColor(cv2.merge([clahe.apply(l), a, b]), cv2.COLOR_LAB2BGR)

# ─────────────────────────────────────────────
# DATASET DISCOVERY
# ─────────────────────────────────────────────
def match_class(folder_name):
    fn = folder_name.lower()
    for canonical, aliases in CLASS_MAP.items():
        for alias in aliases:
            if alias in fn or fn in alias:
                return canonical
    return None

def collect(root, apply_sr_flag, label):
    result = {c: [] for c in CLASS_MAP}
    for d in sorted(root.rglob("*")):
        if not d.is_dir():
            continue
        cls = match_class(d.name)
        if cls is None:
            continue
        imgs = [p for p in d.iterdir() if p.suffix.lower() in IMG_EXTS]
        if imgs:
            result[cls].extend(imgs)
            print(f"  [{label}] {d.relative_to(root)}  →  {cls}  ({len(imgs)} imgs)")
    return result, apply_sr_flag

# ─────────────────────────────────────────────
# PROCESS + SPLIT
# ─────────────────────────────────────────────
def process_image(src, dst, do_sr):
    img = cv2.imread(str(src), cv2.IMREAD_COLOR)
    if img is None:
        img = cv2.imread(str(src), cv2.IMREAD_GRAYSCALE)
    if img is None:
        return False
    if do_sr:
        img = upscale(img)
    img = apply_clahe(img)
    if len(img.shape) == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    img = cv2.resize(img, OUTPUT_SIZE, interpolation=cv2.INTER_AREA)
    cv2.imwrite(str(dst), img)
    return True

def split_and_save(images, class_name, out_root):
    """images = list of (Path, bool:do_sr)"""
    if not images:
        print(f"  [WARN] No images for class '{class_name}'")
        return
    idx = list(range(len(images)))
    tr_idx, tmp = train_test_split(idx, test_size=VAL_RATIO+TEST_RATIO, random_state=SEED)
    vl_idx, te_idx = train_test_split(tmp, test_size=TEST_RATIO/(VAL_RATIO+TEST_RATIO), random_state=SEED)
    for split, idxs in [("train", tr_idx), ("val", vl_idx), ("test", te_idx)]:
        d = out_root / split / class_name
        d.mkdir(parents=True, exist_ok=True)
        print(f"    {split:5s}: {len(idxs):4d} → {d}")
        for i, ix in enumerate(tqdm(idxs, desc=f"  {class_name}/{split}", leave=False)):
            src, do_sr = images[ix]
            process_image(src, d / f"{class_name}_{split}_{i:05d}.jpg", do_sr)

# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────
def main():
    print("=" * 60)
    print("Thermal Dataset Preparation")
    print("=" * 60)
    
    # ─────────────────────────────────────────────
    # CHECK EDSR AVAILABILITY FIRST
    # ─────────────────────────────────────────────
    edsr_available = check_edsr_availability()
    
    if USE_EDSR and not edsr_available:
        print("\n" + "!" * 60)
        print("WARNING: EDSR is enabled but not available!")
        print("The script will fall back to bicubic upscaling (slower and lower quality).")
        print("To fix this, install opencv-contrib-python and/or download the model file.")
        print("!" * 60)
        response = input("\nContinue anyway with bicubic fallback? [y/N]: ").strip().lower()
        if response != 'y':
            print("Aborted by user.")
            return
    elif USE_EDSR and edsr_available:
        print("\n✅ EDSR is ready to use! Will upscale PVF-10 images with EDSR x3.")
    elif not USE_EDSR:
        print("\nℹ️  EDSR is disabled. Will use bicubic upscaling.")
    
    print("\n" + "=" * 60)
    
    for p, n in [(PVF10_ROOT, "PVF-10"), (PVMD_ROOT, "PVMD")]:
        if not p.exists():
            raise FileNotFoundError(f"{n} not found: {p}")

    if OUTPUT_ROOT.exists():
        ans = input(f"\nOutput exists: {OUTPUT_ROOT}\nDelete and recreate? [y/N] ").strip().lower()
        if ans == "y":
            shutil.rmtree(OUTPUT_ROOT)
        else:
            print("Aborted.")
            return

    print("\n[1/3] Discovering images...")
    pvf10, pvf10_sr = collect(PVF10_ROOT, True,  "PVF-10")
    pvmd,  pvmd_sr  = collect(PVMD_ROOT,  False, "PVMD  ")

    merged = {c: [] for c in CLASS_MAP}
    for c in CLASS_MAP:
        merged[c] += [(p, True)  for p in pvf10[c]]   # EDSR ×3
        merged[c] += [(p, False) for p in pvmd[c]]    # no EDSR

    print("\n[2/3] Class totals:")
    for c, imgs in merged.items():
        n_sr = sum(1 for _, sr in imgs if sr)
        n_no = sum(1 for _, sr in imgs if not sr)
        print(f"  {c:15s}  total={len(imgs):4d}  PVF-10(+EDSR)={n_sr}  PVMD={n_no}")

    if all(len(v) == 0 for v in merged.values()):
        print("\n[ERROR] No images collected. Your folder names may not match CLASS_MAP.")
        print("Run these to inspect your actual folder names:")
        print(f"  find '{PVF10_ROOT}' -type d | head -40")
        print(f"  find '{PVMD_ROOT}'  -type d | head -40")
        print("Then update CLASS_MAP at the top of this script accordingly.")
        return

    print(f"\n[3/3] Processing → {OUTPUT_ROOT}")
    print(f"  EDSR: {'enabled (EDSR_x3.pb)' if USE_EDSR else 'disabled → bicubic'}  |  output: {OUTPUT_SIZE}")
    for c in CLASS_MAP:
        print(f"\n  ── {c} ──")
        split_and_save(merged[c], c, OUTPUT_ROOT)

    print("\n" + "=" * 60)
    print("Final counts:")
    for split in ["train", "val", "test"]:
        tot = 0
        for c in CLASS_MAP:
            d = OUTPUT_ROOT / split / c
            n = len(list(d.glob("*.jpg"))) if d.exists() else 0
            tot += n
            print(f"  {split:5s}/{c:15s}: {n}")
        print(f"  {split} total = {tot}\n")

    print(f"Dataset ready at: {OUTPUT_ROOT}")
    print("\nLoad in PyTorch:")
    print("  from torchvision import datasets")
    print("  train_ds = datasets.ImageFolder(root='...thermal_unified/train', transform=...)")

if __name__ == "__main__":
    main()