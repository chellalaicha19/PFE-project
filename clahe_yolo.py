import cv2
from pathlib import Path

def apply_clahe(img_bgr):
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l_clahe = clahe.apply(l)
    lab_clahe = cv2.merge([l_clahe, a, b])
    return cv2.cvtColor(lab_clahe, cv2.COLOR_LAB2BGR)

YOLO_DATASET = Path('rgb/yolo_panel_600_poly_fixed')
OUTPUT = Path('prepared/panel_detection_clahe')

total_images = 0
total_labels = 0
skipped = 0

for split in ['train', 'valid', 'test']:
    img_in  = YOLO_DATASET / split / 'images'
    img_out = OUTPUT / split / 'images'
    lbl_in  = YOLO_DATASET / split / 'labels'
    lbl_out = OUTPUT / split / 'labels'
    img_out.mkdir(parents=True, exist_ok=True)
    lbl_out.mkdir(parents=True, exist_ok=True)

    # Debug: show what's in the folder
    if img_in.exists():
        all_files = list(img_in.iterdir())
        print(f"\n[{split}] Found {len(all_files)} files in {img_in}")
        for f in all_files[:5]:  # show first 5
            print(f"  → {f.name} (suffix: {f.suffix})")
    else:
        print(f"\n[{split}] ⚠️  Folder does not exist: {img_in}")
        continue

    for img_path in img_in.iterdir():
        if img_path.suffix.lower() not in ['.jpg', '.jpeg', '.png']:
            continue

        img = cv2.imread(str(img_path))
        if img is None:
            print(f"  ⚠️  Could not read: {img_path.name}")
            skipped += 1
            continue

        img_clahe = apply_clahe(img)
        cv2.imwrite(str(img_out / img_path.name), img_clahe)
        total_images += 1

        lbl_path = lbl_in / (img_path.stem + '.txt')
        if lbl_path.exists():
            (lbl_out / lbl_path.name).write_bytes(lbl_path.read_bytes())
            total_labels += 1

print(f"\n✅ Done.")
print(f"   Images processed : {total_images}")
print(f"   Labels copied    : {total_labels}")
print(f"   Skipped          : {skipped}")