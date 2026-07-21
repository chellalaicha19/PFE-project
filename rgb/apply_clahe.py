# apply_clahe.py
import cv2
import os
from pathlib import Path

def apply_clahe(src_dir, dst_dir, clip_limit=2.0, tile_grid=(8, 8)):
    dst_dir = Path(dst_dir)
    dst_dir.mkdir(parents=True, exist_ok=True)
    
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid)
    
    exts = {'.jpg', '.jpeg', '.png'}
    files = [f for f in Path(src_dir).iterdir() if f.suffix.lower() in exts]
    print(f"Processing {len(files)} images...")
    
    for fpath in files:
        img = cv2.imread(str(fpath))
        if img is None:
            print(f"  Skipping unreadable: {fpath.name}")
            continue
        
        # Convert to LAB, apply CLAHE only on L channel
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        l_clahe = clahe.apply(l)
        lab_clahe = cv2.merge([l_clahe, a, b])
        result = cv2.cvtColor(lab_clahe, cv2.COLOR_LAB2BGR)
        
        out_path = dst_dir / fpath.name
        cv2.imwrite(str(out_path), result)
    
    print(f"Done! CLAHE images saved to: {dst_dir}")

# Run on train and val
apply_clahe(
    "Panel_Detection.yolov8-obb/train/images",
    "Panel_Detection.yolov8-obb/train/images_clahe"
)