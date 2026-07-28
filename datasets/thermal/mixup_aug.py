"""
Contrast-Preserving MixUp Augmentation for Thermal Fault Classes
================================================================
Applies MixUp only within fault classes (Hotspot, Offline_Cold_Module,
String_Interconnect_Failure), preserving thermal contrast gradients.

Usage:
    python thermal_mixup_augment.py \
        --input_dir prepared/thermal_dataset \
        --output_dir prepared/thermal_augmented \
        --target_count 3000   # target images per fault class
"""

import os
import cv2
import numpy as np
from pathlib import Path
import argparse
import random
from tqdm import tqdm

# Classes to augment (exclude No_Anomaly)
FAULT_CLASSES = ["Hotspot", "Offline_Cold_Module", "String_Interconnect_Failure"]

LAMBDA_MIN = 0.35   # safe mixing ratio range — avoids destroying thermal contrast
LAMBDA_MAX = 0.65


def contrast_preserving_mixup(img1: np.ndarray, img2: np.ndarray) -> np.ndarray:
    """
    MixUp with post-blend contrast optimisation.
    Tries N random lambda values and picks the one that maximises
    the std of pixel intensities (= sharpest thermal gradients).
    """
    best_img = None
    best_std = -1.0

    for _ in range(8):  # sample 8 candidate lambdas
        lam = random.uniform(LAMBDA_MIN, LAMBDA_MAX)
        blended = (lam * img1.astype(np.float32) +
                   (1 - lam) * img2.astype(np.float32))
        blended = np.clip(blended, 0, 255)
        std = blended.std()
        if std > best_std:
            best_std = std
            best_img = blended

    return best_img.astype(np.uint8)


def load_images_with_names(class_dir: Path) -> list[tuple[np.ndarray, str]]:
    """Load images with their filenames (without extension)."""
    images_with_names = []
    exts = {".jpg", ".jpeg", ".png", ".bmp"}
    for f in class_dir.iterdir():
        if f.suffix.lower() in exts:
            img = cv2.imread(str(f))
            if img is not None:
                name = f.stem  # get filename without extension
                images_with_names.append((img, name))
    return images_with_names


def augment_class(class_name: str, input_dir: Path, output_dir: Path,
                  target_count: int):
    src_dir = input_dir / class_name
    dst_dir = output_dir / class_name
    dst_dir.mkdir(parents=True, exist_ok=True)

    # Load originals with their names
    originals_with_names = load_images_with_names(src_dir)
    originals = [img for img, _ in originals_with_names]
    original_names = [name for _, name in originals_with_names]

    # Copy originals first
    for f in src_dir.iterdir():
        if f.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}:
            img = cv2.imread(str(f))
            if img is not None:
                out_path = dst_dir / f.name
                cv2.imwrite(str(out_path), img)

    existing_count = len(originals)
    needed = max(0, target_count - existing_count)

    if needed == 0:
        print(f"[{class_name}] Already {existing_count} images — no augmentation needed.")
        return

    print(f"[{class_name}] {existing_count} originals → generating {needed} MixUp samples "
          f"(target: {target_count})")

    h, w = originals[0].shape[:2]

    for i in tqdm(range(needed), desc=f"  MixUp {class_name}"):
        # Sample two different images with their names
        idx1, idx2 = random.sample(range(len(originals)), 2)
        img1 = originals[idx1]
        img2 = originals[idx2]
        name1 = original_names[idx1]
        name2 = original_names[idx2]
        
        # Resize to same shape if needed
        if img2.shape[:2] != (h, w):
            img2 = cv2.resize(img2, (w, h))

        mixed = contrast_preserving_mixup(img1, img2)
        
        # Create filename: aug_mixup_original1_original2.jpg
        out_name = f"aug_mixup_{name1}_{name2}.jpg"
        cv2.imwrite(str(dst_dir / out_name), mixed)

    print(f"[{class_name}] Done. Total: {existing_count + needed} images.")


def copy_no_anomaly(input_dir: Path, output_dir: Path):
    """Copy No_Anomaly as-is (no augmentation needed)."""
    src = input_dir / "No_Anomaly"
    dst = output_dir / "No_Anomaly"
    dst.mkdir(parents=True, exist_ok=True)
    count = 0
    for f in src.iterdir():
        if f.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}:
            img = cv2.imread(str(f))
            if img is not None:
                cv2.imwrite(str(dst / f.name), img)
                count += 1
    print(f"[No_Anomaly] Copied {count} images unchanged.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir",  default="prepared/thermal_dataset_sr_96x72")
    parser.add_argument("--output_dir", default="prepared/thermal_augmented")
    parser.add_argument("--target_count", type=int, default=4000,
                        help="Target number of images per fault class after augmentation")
    args = parser.parse_args()

    input_dir  = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Copy No_Anomaly unchanged
    copy_no_anomaly(input_dir, output_dir)

    # 2. Augment each fault class with contrast-preserving MixUp
    for cls in FAULT_CLASSES:
        augment_class(cls, input_dir, output_dir, args.target_count)

    print("\nMixUp augmentation complete.")
    print(f"Output: {output_dir}")


if __name__ == "__main__":
    main()