import cv2
import numpy as np
from ultralytics import YOLO
from pathlib import Path

# ── CONFIG ────────────────────────────────────────────────────────────────────
MODEL_PATH   = '/Users/mac/Documents/PFE/models.pt/best_clahe_600.pt'  # <-- USE YOUR TRAINED WEIGHTS
IMAGES_DIR   = '/Users/mac/Documents/PFE/prepared/test2'
OUTPUT_DIR   = '/Users/mac/Documents/PFE/prepared/test_results'
CONF_THRESH  = 0.5

# ── CLAHE (match training preprocessing) ──────────────────────────────────────
def apply_clahe(img_bgr):
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l_clahe = clahe.apply(l)
    lab_clahe = cv2.merge([l_clahe, a, b])
    return cv2.cvtColor(lab_clahe, cv2.COLOR_LAB2BGR)

# ── Setup ─────────────────────────────────────────────────────────────────────
Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
model = YOLO(MODEL_PATH)

image_paths = list(Path(IMAGES_DIR).glob('*.[pP][nN][gG]*'))
print(f"Found {len(image_paths)} images\n")

# ── Run inference ─────────────────────────────────────────────────────────────
for img_path in image_paths:
    img = cv2.imread(str(img_path))
    if img is None:
        print(f"  ⚠️  Could not read {img_path.name}")
        continue

    # Apply CLAHE
    img_clahe = apply_clahe(img)

    # Run YOLO OBB detection
    results = model(img_clahe, conf=CONF_THRESH, verbose=False)[0]

    # Check for OBB boxes (rotated rectangles)
    if results.obb is not None and len(results.obb) > 0:
        boxes = results.obb
        print(f"  {img_path.name}: {len(boxes)} panel(s) detected (OBB)")
        
        annotated = img_clahe.copy()
        
        for box in boxes:
            # OBB format: get 4 corner points
            xyxyxyxy = box.xyxyxyxy[0].cpu().numpy().reshape(-1, 2).astype(int)
            conf = float(box.conf[0])
            cls = int(box.cls[0])
            
            # Draw rotated polygon
            cv2.polylines(annotated, [xyxyxyxy], True, (0, 255, 0), 2)
            
            # Add label at first corner
            label = f'panel {conf:.2f}'
            cv2.putText(annotated, label, (xyxyxyxy[0][0], xyxyxyxy[0][1] - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            
            # Get axis-aligned crop for saving (simpler)
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            h, w = img_clahe.shape[:2]
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)
            
            if x2 > x1 and y2 > y1:
                crop = img_clahe[y1:y2, x1:x2]
                crop_name = f'{img_path.stem}_crop_{x1}_{y1}.jpg'
                cv2.imwrite(str(Path(OUTPUT_DIR) / crop_name), crop)

        # Save annotated image
        out_path = Path(OUTPUT_DIR) / f'detected_{img_path.name}'
        cv2.imwrite(str(out_path), annotated)
        
    elif results.boxes is not None and len(results.boxes) > 0:
        # Fallback to regular boxes if OBB not available
        boxes = results.boxes
        print(f"  {img_path.name}: {len(boxes)} panel(s) detected (standard boxes)")
        
        annotated = img_clahe.copy()
        
        for box in boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = float(box.conf[0])
            
            cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)
            label = f'panel {conf:.2f}'
            cv2.putText(annotated, label, (x1, y1 - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            
            crop = img_clahe[y1:y2, x1:x2]
            crop_name = f'{img_path.stem}_cropOBB_{x1}_{y1}.jpg'
            cv2.imwrite(str(Path(OUTPUT_DIR) / crop_name), crop)

        out_path = Path(OUTPUT_DIR) / f'detected_{img_path.name}'
        cv2.imwrite(str(out_path), annotated)
        
    else:
        print(f"  {img_path.name}: No panels detected")
        # Save original CLAHE image even if no detection
        out_path = Path(OUTPUT_DIR) / f'nodetect_{img_path.name}'
        cv2.imwrite(str(out_path), img_clahe)

print(f'\n✅ Done. Results saved to: {OUTPUT_DIR}')