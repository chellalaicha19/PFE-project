#!/usr/bin/env python3
"""
Map YOLO images to class folders based on highest priority class in labels
Priority order: 0 < 1 < 5 < 6 (higher number = higher priority for mapping)
Ignores train/test/valid splits, just organizes by class
"""

import os
import shutil
import glob
from collections import defaultdict

# Configuration
SOURCE_BASE = "/Users/mac/Documents/PFE/rgb/solar_panel_fault_detection.v1i.yolov8"
DEST_BASE = "/Users/mac/Documents/PFE/unified_dataset"

# Class priority (higher value = higher priority for final class)
# 6 is highest priority, 0 is lowest
CLASS_PRIORITY = {
    0: 0,  # Lowest priority
    1: 1,
    5: 2,
    6: 3   # Highest priority
}

# Target class folders (as they exist in unified_dataset)
TARGET_FOLDERS = {
    0: "Class_0_Clean_panels",
    1: "Class_1_Soiling_pollution",
    2: "Class_2_Shadowing_vegetation",
    5: "Class_5_Cell_string_failure",
    6: "Class_6_Structural_damage"
}

# Valid target classes
VALID_CLASSES = {0, 1, 2, 5, 6}

def get_all_classes_from_label(label_path):
    """Extract all unique class IDs from a label file"""
    classes = set()
    try:
        with open(label_path, 'r') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#'):
                    parts = line.split()
                    if len(parts) >= 5:  # YOLO format
                        classes.add(int(parts[0]))
    except Exception as e:
        print(f"  Error reading {label_path}: {e}")
        return None
    return classes

def get_highest_priority_class(classes):
    """Get the class with highest priority (largest priority value)"""
    if not classes:
        return None
    
    # Filter to only valid classes
    valid_classes = [c for c in classes if c in VALID_CLASSES]
    
    if not valid_classes:
        return None
    
    # Return class with highest priority value
    return max(valid_classes, key=lambda c: CLASS_PRIORITY.get(c, -1))

def find_image_file(image_dir, basename):
    """Find image file with various extensions"""
    for ext in ['.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG']:
        img_path = os.path.join(image_dir, basename + ext)
        if os.path.exists(img_path):
            return img_path
    return None

def process_split(split_name, split_path, stats):
    """Process all images in a split (train/test/valid)"""
    labels_dir = os.path.join(split_path, "labels")  # Use labels, not labels_backup
    images_dir = os.path.join(split_path, "images")
    
    if not os.path.exists(labels_dir):
        print(f"  ⚠️  Labels directory not found: {labels_dir}")
        return
    
    if not os.path.exists(images_dir):
        print(f"  ⚠️  Images directory not found: {images_dir}")
        return
    
    label_files = glob.glob(os.path.join(labels_dir, "*.txt"))
    
    if not label_files:
        print(f"  ⚠️  No label files found in {labels_dir}")
        return
    
    print(f"  Found {len(label_files)} label files")
    
    split_stats = defaultdict(int)
    multi_class_count = 0
    errors = 0
    no_valid_class = 0
    no_image = 0
    
    for label_path in label_files:
        basename = os.path.basename(label_path)
        name_no_ext = os.path.splitext(basename)[0]
        
        # Get all classes from label
        classes_in_label = get_all_classes_from_label(label_path)
        
        if classes_in_label is None:
            print(f"    ⚠️  Error reading: {basename}")
            errors += 1
            continue
        
        if len(classes_in_label) == 0:
            print(f"    ⚠️  No valid classes in: {basename}")
            no_valid_class += 1
            errors += 1
            continue
        
        # Determine target class based on highest priority
        target_class = get_highest_priority_class(classes_in_label)
        
        if target_class is None:
            print(f"    ⚠️  No valid target class for: {basename} (classes: {classes_in_label})")
            no_valid_class += 1
            errors += 1
            continue
        
        # Track multi-class images
        if len(classes_in_label) > 1:
            multi_class_count += 1
            print(f"    ℹ️  Multi-class: {basename} - classes: {classes_in_label} -> selected class {target_class}")
        
        # Find corresponding image
        img_path = find_image_file(images_dir, name_no_ext)
        
        if not img_path:
            print(f"    ⚠️  Image not found for: {basename}")
            no_image += 1
            errors += 1
            continue
        
        # Destination path
        target_folder = TARGET_FOLDERS[target_class]
        dest_dir = os.path.join(DEST_BASE, target_folder)
        dest_path = os.path.join(dest_dir, os.path.basename(img_path))
        
        # Handle duplicate filenames
        if os.path.exists(dest_path):
            name, ext = os.path.splitext(os.path.basename(img_path))
            counter = 1
            while os.path.exists(os.path.join(dest_dir, f"{name}_{counter}{ext}")):
                counter += 1
            dest_path = os.path.join(dest_dir, f"{name}_{counter}{ext}")
        
        # Copy the image
        try:
            shutil.copy2(img_path, dest_path)
            stats[target_class] += 1
            split_stats[target_class] += 1
        except Exception as e:
            print(f"    ❌ Error copying {img_path}: {e}")
            errors += 1
    
    # Print split summary
    if split_stats:
        print(f"\n  📊 {split_name.upper()} split results:")
        for cls in sorted(split_stats.keys()):
            class_name = TARGET_FOLDERS[cls]
            print(f"    {class_name}: {split_stats[cls]} images")
        
        if multi_class_count > 0:
            print(f"\n  📝 Multi-class images: {multi_class_count}")
    
    if errors > 0:
        print(f"\n  ⚠️  Errors in {split_name}: {errors}")
        print(f"    - No valid class: {no_valid_class}")
        print(f"    - Image not found: {no_image}")

def verify_destination_folders():
    """Verify that all destination folders exist"""
    missing_folders = []
    for folder in TARGET_FOLDERS.values():
        folder_path = os.path.join(DEST_BASE, folder)
        if not os.path.exists(folder_path):
            missing_folders.append(folder)
        else:
            print(f"✅ Found: {folder}")
    
    if missing_folders:
        print("\n❌ Missing destination folders:")
        for folder in missing_folders:
            print(f"   - {folder}")
        print("\nPlease create these folders in unified_dataset first!")
        return False
    return True

def main():
    print("=" * 70)
    print("YOLO Dataset Mapper - Priority-Based Class Selection")
    print("=" * 70)
    print(f"\nSource: {SOURCE_BASE}")
    print(f"Destination: {DEST_BASE}")
    
    print("\nClass Priority Order (higher number = higher priority):")
    print("  6 (Structural damage) > 5 (Cell failure) > 1 (Soiling) > 0 (Clean)")
    
    print("\nMapping Rules:")
    print("  - Images with multiple classes → placed in highest priority class")
    print("  - All splits (train/test/valid) combined → organized by class only")
    print("  - Only images copied (no labels)")
    
    # Verify destination folders
    print("\n" + "=" * 70)
    print("VERIFYING DESTINATION FOLDERS")
    print("=" * 70)
    
    if not verify_destination_folders():
        return
    
    confirm = input("\nProceed with mapping? (yes/no): ").lower().strip()
    if confirm != 'yes':
        print("\n❌ Cancelled.")
        return
    
    print("\n" + "=" * 70)
    print("PROCESSING...")
    print("=" * 70)
    
    # Process each split
    splits = ['train', 'test', 'valid']
    total_stats = defaultdict(int)
    
    for split in splits:
        split_path = os.path.join(SOURCE_BASE, split)
        if os.path.exists(split_path):
            print(f"\n📁 Processing {split.upper()} split...")
            print("-" * 50)
            process_split(split, split_path, total_stats)
        else:
            print(f"\n⚠️  Split directory not found: {split_path}")
    
    # Final summary
    print("\n" + "=" * 70)
    print("FINAL SUMMARY")
    print("=" * 70)
    
    print("\nTotal images per class:")
    for cls in sorted(total_stats.keys()):
        folder = TARGET_FOLDERS[cls]
        count = total_stats[cls]
        print(f"  {folder}: {count} images")
    
    total_images = sum(total_stats.values())
    print(f"\n📊 TOTAL IMAGES PROCESSED: {total_images}")
    
    # Show final counts in destination folders
    print("\n✅ Final counts in unified_dataset:")
    for cls in sorted(TARGET_FOLDERS.keys()):
        folder = TARGET_FOLDERS[cls]
        folder_path = os.path.join(DEST_BASE, folder)
        if os.path.exists(folder_path):
            img_count = len([f for f in os.listdir(folder_path) 
                           if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
            print(f"  📁 {folder}: {img_count} images")
    
    print("\n✅ Done! Images organized by highest priority class.")
    print("   No labels were copied, only images.")

if __name__ == "__main__":
    main()