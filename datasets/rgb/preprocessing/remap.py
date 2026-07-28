#!/usr/bin/env python3
"""
Backup and remap YOLO label class numbers
Original classes: 0,1,2,3,4,5 -> New classes: 1,1,1,5,0,6
"""

import os
import shutil
import glob
from pathlib import Path

# Configuration
SOURCE_BASE = "/Users/mac/Documents/PFE/rgb/solar_panel_fault_detection.v1i.yolov8"
BACKUP_SUFFIX = "_backup"  # Will create labels_backup folders

# Class mapping: Original class -> New class
CLASS_MAPPING = {
    0: 1,  # 0 becomes 1
    1: 1,  # 1 becomes 1
    2: 1,  # 2 becomes 1
    3: 5,  # 3 becomes 5
    4: 0,  # 4 becomes 0
    5: 6,  # 5 becomes 6
}

def backup_labels(split_path):
    """Create backup of all label files"""
    labels_dir = os.path.join(split_path, "labels")
    backup_dir = os.path.join(split_path, "labels_backup")
    
    if not os.path.exists(labels_dir):
        print(f"  ⚠️  Labels directory not found: {labels_dir}")
        return False
    
    # Create backup directory if it doesn't exist
    os.makedirs(backup_dir, exist_ok=True)
    
    # Copy all label files to backup
    label_files = glob.glob(os.path.join(labels_dir, "*.txt"))
    
    if not label_files:
        print(f"  ⚠️  No label files found in {labels_dir}")
        return False
    
    backed_up = 0
    for label_file in label_files:
        filename = os.path.basename(label_file)
        backup_path = os.path.join(backup_dir, filename)
        try:
            shutil.copy2(label_file, backup_path)
            backed_up += 1
        except Exception as e:
            print(f"    ❌ Error backing up {filename}: {e}")
    
    print(f"  ✅ Backed up {backed_up} label files to {backup_dir}")
    return True

def remap_label_file(label_path, mapping):
    """Remap class numbers in a single label file"""
    try:
        # Read all lines
        with open(label_path, 'r') as f:
            lines = f.readlines()
        
        # Process each line
        new_lines = []
        modified = False
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
            
            parts = line.split()
            if len(parts) >= 5:  # YOLO format: class x_center y_center width height
                original_class = int(parts[0])
                
                # Apply mapping if exists
                if original_class in mapping:
                    new_class = mapping[original_class]
                    parts[0] = str(new_class)
                    modified = True
                    print(f"      Class {original_class} -> {new_class}")
                
                new_lines.append(' '.join(parts))
        
        # Write back if modified
        if modified:
            with open(label_path, 'w') as f:
                for line in new_lines:
                    f.write(line + '\n')
            return True
        return False
        
    except Exception as e:
        print(f"    ❌ Error processing {label_path}: {e}")
        return False

def process_split(split_name, split_path):
    """Process train/test/valid split"""
    print(f"\n📁 Processing {split_name.upper()} split...")
    print("-" * 50)
    
    labels_dir = os.path.join(split_path, "labels")
    
    if not os.path.exists(labels_dir):
        print(f"  ⚠️  Labels directory not found: {labels_dir}")
        return {'backed_up': 0, 'remapped': 0, 'errors': 0}
    
    # Step 1: Backup labels
    print("  Step 1: Creating backup...")
    backup_labels(split_path)
    
    # Step 2: Remap class numbers
    print("  Step 2: Remapping class numbers...")
    label_files = glob.glob(os.path.join(labels_dir, "*.txt"))
    
    remapped_count = 0
    error_count = 0
    modified_files = 0
    
    for label_file in label_files:
        filename = os.path.basename(label_file)
        print(f"    Processing: {filename}")
        
        if remap_label_file(label_file, CLASS_MAPPING):
            modified_files += 1
            remapped_count += 1
        else:
            error_count += 1
    
    print(f"\n  📊 {split_name.upper()} results:")
    print(f"    ✅ Modified files: {modified_files}")
    print(f"    📝 Total labels processed: {remapped_count}")
    if error_count > 0:
        print(f"    ❌ Errors: {error_count}")
    
    return {
        'backed_up': len(glob.glob(os.path.join(split_path, "labels_backup", "*.txt"))),
        'remapped': modified_files,
        'errors': error_count
    }

def verify_mapping(split_path):
    """Verify the mapping by checking a few random files"""
    labels_dir = os.path.join(split_path, "labels")
    backup_dir = os.path.join(split_path, "labels_backup")
    
    if not os.path.exists(labels_dir) or not os.path.exists(backup_dir):
        return
    
    label_files = glob.glob(os.path.join(labels_dir, "*.txt"))
    
    if not label_files:
        return
    
    print("\n  Verification (first 3 files):")
    for label_file in label_files[:3]:
        filename = os.path.basename(label_file)
        backup_file = os.path.join(backup_dir, filename)
        
        if os.path.exists(backup_file):
            # Read original first line
            with open(backup_file, 'r') as f:
                original_line = f.readline().strip()
            
            # Read new first line
            with open(label_file, 'r') as f:
                new_line = f.readline().strip()
            
            if original_line and new_line:
                original_class = original_line.split()[0] if original_line else 'N/A'
                new_class = new_line.split()[0] if new_line else 'N/A'
                print(f"    {filename}: Class {original_class} -> {new_class}")

def main():
    print("=" * 70)
    print("YOLO Label Backup and Remapper")
    print("=" * 70)
    print(f"\nSource: {SOURCE_BASE}")
    print("\nClass Mapping:")
    for orig, new in sorted(CLASS_MAPPING.items()):
        print(f"  Original class {orig} -> New class {new}")
    
    print("\nThis will:")
    print("  1. Create labels_backup folders in train/test/valid")
    print("  2. Copy all original labels to backup folders")
    print("  3. Modify class numbers in original label files")
    print("  4. Keep the same folder structure")
    
    confirm = input("\nProceed? (yes/no): ").lower().strip()
    if confirm != 'yes':
        print("\n❌ Cancelled.")
        return
    
    print("\n" + "=" * 70)
    print("PROCESSING...")
    print("=" * 70)
    
    # Process each split
    splits = ['train', 'test', 'valid']
    total_stats = {
        'backed_up': 0,
        'remapped': 0,
        'errors': 0
    }
    
    for split in splits:
        split_path = os.path.join(SOURCE_BASE, split)
        if os.path.exists(split_path):
            stats = process_split(split, split_path)
            total_stats['backed_up'] += stats['backed_up']
            total_stats['remapped'] += stats['remapped']
            total_stats['errors'] += stats['errors']
        else:
            print(f"\n⚠️  Split directory not found: {split_path}")
    
    # Verification
    print("\n" + "=" * 70)
    print("VERIFICATION")
    print("=" * 70)
    
    for split in splits:
        split_path = os.path.join(SOURCE_BASE, split)
        if os.path.exists(split_path):
            print(f"\n{split.upper()} split:")
            verify_mapping(split_path)
    
    # Final summary
    print("\n" + "=" * 70)
    print("FINAL SUMMARY")
    print("=" * 70)
    print(f"\n✅ Backed up: {total_stats['backed_up']} label files")
    print(f"✅ Remapped: {total_stats['remapped']} label files")
    if total_stats['errors'] > 0:
        print(f"⚠️  Errors: {total_stats['errors']}")
    
    print("\n📁 New folder structure:")
    for split in splits:
        split_path = os.path.join(SOURCE_BASE, split)
        if os.path.exists(split_path):
            labels_backup = os.path.join(split_path, "labels_backup")
            if os.path.exists(labels_backup):
                backup_count = len(glob.glob(os.path.join(labels_backup, "*.txt")))
                print(f"  {split}/labels_backup/ ({backup_count} backup files)")
    
    print("\n✅ Done! Labels have been backed up and remapped.")
    print("   Original labels are safe in 'labels_backup' folders")

if __name__ == "__main__":
    main()