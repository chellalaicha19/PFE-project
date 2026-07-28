import json
from pathlib import Path
import shutil
from collections import Counter

# ==================== CONFIG ====================
JSON_PATH = Path("/Users/mac/Documents/PFE/thermal/InfraredSolarModules/module_metadata.json")          # your attached file
IMAGES_DIR = Path("/Users/mac/Documents/PFE/thermal/InfraredSolarModules/images")                       # folder where all .jpg files live
OUTPUT_BASE = Path("prepared/thermal_dataset")             # new folder that will contain the 4 classes

# Exact mapping from the paper's 12 classes → your 4 target classes
# (this is the same grouping I suggested earlier)
CLASS_MAPPING = {
    "No-Anomaly":         "No_Anomaly",
    "Cell":               "Hotspot",
    "Cell-Multi":         "Hotspot",
    "Cracking":           "Hotspot",
    "Hot-Spot":           "Hotspot",
    "Hot-Spot-Multi":     "Hotspot",
    "Diode":              "String_Interconnect_Failure",
    "Diode-Multi":        "String_Interconnect_Failure",
    "Offline-Module":     "Offline_Cold_Module",
    "Shadowing":          "Offline_Cold_Module",
    "Vegetation":         "Offline_Cold_Module",
    "Soiling":            "Offline_Cold_Module",
}

# ===============================================

# 1. Load the JSON
print("Loading metadata...")
with open(JSON_PATH, encoding="utf-8") as f:
    metadata = json.load(f)

print(f"Total images in JSON: {len(metadata):,}")

# 2. Create the 4 class folders
OUTPUT_BASE.mkdir(parents=True, exist_ok=True)

class_folders = {}
for new_class in set(CLASS_MAPPING.values()):
    folder = OUTPUT_BASE / new_class
    folder.mkdir(parents=True, exist_ok=True)
    class_folders[new_class] = folder

# 3. Remap and copy images + count
print("Copying images into 4 classes...")

counts = Counter()
copied = 0
skipped = 0

for img_id, info in metadata.items():
    old_class = info["anomaly_class"]
    img_path = IMAGES_DIR / info["image_filepath"].split("/")[-1]   # e.g. "images/13357.jpg" → just the filename

    if old_class not in CLASS_MAPPING:
        print(f"⚠️  Unknown class '{old_class}' for image {img_id} — skipping")
        skipped += 1
        continue

    new_class_name = CLASS_MAPPING[old_class]
    dest_folder = class_folders[new_class_name]
    dest_path = dest_folder / img_path.name

    if img_path.exists():
        shutil.copy2(img_path, dest_path)
        counts[new_class_name] += 1
        copied += 1
    else:
        print(f"⚠️  Missing file: {img_path} — skipping")
        skipped += 1

# 4. Final report
print("\n✅ DONE! Dataset remapped to 4 classes.")
print("Folder structure created at:", OUTPUT_BASE.resolve())
for cls, count in sorted(counts.items()):
    print(f"   • {cls:30} → {count:6,} images")

print(f"\nTotal copied : {copied:,}")
print(f"Skipped      : {skipped:,}")

# Optional: save a small summary JSON so you can check later
summary = {
    "total_images": len(metadata),
    "class_counts": dict(counts),
    "mapping_used": CLASS_MAPPING
}
with open(OUTPUT_BASE / "class_summary.json", "w", encoding="utf-8") as f:
    json.dump(summary, f, indent=2)