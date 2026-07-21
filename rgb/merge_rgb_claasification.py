import os
import shutil
from pathlib import Path

# Define the source datasets
SOURCE_DATASETS = {
    'detect_solar_dust': '/Users/mac/Documents/PFE/rgb/Detect_solar_dust',
    'faulty_solar_panel': '/Users/mac/Documents/PFE/rgb/Faulty_solar_panel'
}

# Define the target unified dataset directory
TARGET_DIR = '/Users/mac/Documents/PFE/unified_dataset'

# Define class mapping
# Format: (source_dataset, source_subfolder_or_pattern) -> target_class
CLASS_MAPPING = {
    # Class 0 — Clean panels
    ('detect_solar_dust', 'Clean'): 'Class_0_Clean_panels',
    ('faulty_solar_panel', 'Clean'): 'Class_0_Clean_panels',
    
    # Class 1 — Soiling / pollution
    ('detect_solar_dust', 'Dusty'): 'Class_1_Soiling_pollution',
    ('faulty_solar_panel', 'Dusty'): 'Class_1_Soiling_pollution',
    ('detect_solar_dust', 'Bird-drop'): 'Class_1_Soiling_pollution',
    ('faulty_solar_panel', 'Bird-drop'): 'Class_1_Soiling_pollution',
    
    # Class 2 — Shadowing / vegetation
    ('faulty_solar_panel', 'Snow-Covered'): 'Class_2_Shadowing_vegetation',
    ('detect_solar_dust', 'Snow-Covered'): 'Class_2_Shadowing_vegetation',  # if exists
    ('faulty_solar_panel', 'Vegetation'): 'Class_2_Shadowing_vegetation',
    ('detect_solar_dust', 'Vegetation'): 'Class_2_Shadowing_vegetation',  # if exists
    
    # Class 5 — Cell / string failure
    ('faulty_solar_panel', 'Electrical-damage'): 'Class_5_Cell_string_failure',
    ('detect_solar_dust', 'Electrical-damage'): 'Class_5_Cell_string_failure',  # if exists
    
    # Class 6 — Structural damage
    ('faulty_solar_panel', 'Physical-Damage'): 'Class_6_Structural_damage',
    ('detect_solar_dust', 'Physical-Damage'): 'Class_6_Structural_damage',  # if exists
}

def create_target_directories():
    """Create target class directories"""
    class_dirs = [
        'Class_0_Clean_panels',
        'Class_1_Soiling_pollution', 
        'Class_2_Shadowing_vegetation',
        'Class_5_Cell_string_failure',
        'Class_6_Structural_damage'
    ]
    
    for class_dir in class_dirs:
        path = os.path.join(TARGET_DIR, class_dir)
        os.makedirs(path, exist_ok=True)
        print(f"Created directory: {path}")
    
    return class_dirs

def copy_files(src_path, dst_path, filename):
    """Copy a file from source to destination"""
    src_file = os.path.join(src_path, filename)
    dst_file = os.path.join(dst_path, filename)
    
    # Handle duplicate filenames by adding a prefix
    if os.path.exists(dst_file):
        name, ext = os.path.splitext(filename)
        counter = 1
        while os.path.exists(os.path.join(dst_path, f"{name}_{counter}{ext}")):
            counter += 1
        dst_file = os.path.join(dst_path, f"{name}_{counter}{ext}")
    
    try:
        shutil.copy2(src_file, dst_file)
        return True
    except Exception as e:
        print(f"  Error copying {src_file}: {e}")
        return False

def process_datasets():
    """Main function to process and map datasets"""
    
    print("=" * 60)
    print("Starting dataset mapping process...")
    print("=" * 60)
    
    # Create target directories
    create_target_directories()
    
    # Statistics
    stats = {class_name: 0 for class_name in set(CLASS_MAPPING.values())}
    errors = []
    
    # Process each mapping rule
    for (dataset, source_class), target_class in CLASS_MAPPING.items():
        source_path = SOURCE_DATASETS.get(dataset)
        
        if not source_path:
            print(f"\n⚠️  Warning: Dataset '{dataset}' not found in SOURCE_DATASETS")
            continue
        
        source_class_path = os.path.join(source_path, source_class)
        
        if not os.path.exists(source_class_path):
            print(f"\n⚠️  Source path not found: {source_class_path}")
            continue
        
        print(f"\n📁 Processing: {dataset}/{source_class} -> {target_class}")
        
        # Get all files in the source class directory
        try:
            files = os.listdir(source_class_path)
            image_files = [f for f in files if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff'))]
            
            if not image_files:
                print(f"  ⚠️  No image files found in {source_class_path}")
                continue
            
            target_path = os.path.join(TARGET_DIR, target_class)
            copied_count = 0
            
            for filename in image_files:
                if copy_files(source_class_path, target_path, filename):
                    copied_count += 1
                    stats[target_class] += 1
            
            print(f"  ✅ Copied {copied_count} images from {source_class}")
            
        except Exception as e:
            error_msg = f"Error processing {source_class_path}: {e}"
            print(f"  ❌ {error_msg}")
            errors.append(error_msg)
    
    # Print summary
    print("\n" + "=" * 60)
    print("MAPPING COMPLETE - SUMMARY")
    print("=" * 60)
    
    total_images = 0
    for class_name, count in stats.items():
        if count > 0:
            print(f"{class_name}: {count} images")
            total_images += count
    
    print(f"\n📊 Total images processed: {total_images}")
    
    # Print target directory structure
    print("\n📁 Target directory structure:")
    for root, dirs, files in os.walk(TARGET_DIR):
        level = root.replace(TARGET_DIR, '').count(os.sep)
        indent = '  ' * level
        print(f"{indent}{os.path.basename(root)}/")
        if level == 1:  # Only show immediate subdirectories
            subindent = '  ' * (level + 1)
            print(f"{subindent}({len(files)} files)")
    
    if errors:
        print(f"\n⚠️  Errors encountered: {len(errors)}")
        for error in errors[:5]:  # Show first 5 errors
            print(f"  - {error}")
    
    print("\n✅ Dataset mapping completed!")

def verify_structure():
    """Verify the final directory structure"""
    print("\n" + "=" * 60)
    print("VERIFYING DIRECTORY STRUCTURE")
    print("=" * 60)
    
    expected_classes = [
        'Class_0_Clean_panels',
        'Class_1_Soiling_pollution',
        'Class_2_Shadowing_vegetation',
        'Class_5_Cell_string_failure',
        'Class_6_Structural_damage'
    ]
    
    for class_name in expected_classes:
        class_path = os.path.join(TARGET_DIR, class_name)
        if os.path.exists(class_path):
            file_count = len([f for f in os.listdir(class_path) 
                            if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff'))])
            print(f"✅ {class_name}: {file_count} images")
        else:
            print(f"❌ {class_name}: Directory not found")

if __name__ == "__main__":
    # Run the mapping process
    process_datasets()
    
    # Verify the final structure
    verify_structure()
    
    print("\n💡 Tip: To check the contents of any class, run:")
    print("  ls -la /Users/mac/Documents/PFE/unified_dataset/Class_0_Clean_panels/ | head")