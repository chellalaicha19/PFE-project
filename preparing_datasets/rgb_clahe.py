import cv2
import os
import numpy as np
from pathlib import Path
from tqdm import tqdm
import argparse

def apply_clahe_to_image(image, clip_limit=2.0, tile_grid_size=(8, 8)):
    """
    Apply CLAHE to an image using LAB color space.
    
    Args:
        image: BGR image from cv2.imread()
        clip_limit: Contrast limiting threshold
        tile_grid_size: Size of grid for histogram equalization
    
    Returns:
        Image with CLAHE applied
    """
    # Create CLAHE object
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    
    # Convert BGR to LAB color space
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    
    # Split LAB channels
    l, a, b = cv2.split(lab)
    
    # Apply CLAHE to L channel
    l_clahe = clahe.apply(l)
    
    # Merge channels back
    lab_clahe = cv2.merge([l_clahe, a, b])
    
    # Convert back to BGR
    image_clahe = cv2.cvtColor(lab_clahe, cv2.COLOR_LAB2BGR)
    
    return image_clahe

def resize_and_process_image(input_path, output_path, target_size=(640, 640), 
                             clip_limit=2.0, tile_grid_size=(8, 8)):
    """
    Resize image and apply CLAHE.
    
    Args:
        input_path: Path to input image
        output_path: Path to save processed image
        target_size: Target size (width, height)
        clip_limit: CLAHE clip limit
        tile_grid_size: CLAHE tile grid size
    """
    # Read image
    img = cv2.imread(str(input_path))
    if img is None:
        print(f"Warning: Could not read {input_path}")
        return False
    
    try:
        # Resize image to target size
        img_resized = cv2.resize(img, target_size, interpolation=cv2.INTER_LANCZOS4)
        
        # Apply CLAHE
        img_processed = apply_clahe_to_image(img_resized, clip_limit, tile_grid_size)
        
        # Save processed image (preserve original format)
        cv2.imwrite(str(output_path), img_processed)
        return True
        
    except Exception as e:
        print(f"Error processing {input_path}: {e}")
        return False

def process_folder(input_folder, output_folder, target_size=(640, 640),
                   clip_limit=2.0, tile_grid_size=(8, 8), 
                   extensions=None, preserve_structure=False):
    """
    Process all images in a folder.
    
    Args:
        input_folder: Folder containing input images
        output_folder: Folder to save processed images
        target_size: Target size tuple (width, height)
        clip_limit: CLAHE clip limit
        tile_grid_size: CLAHE tile grid size
        extensions: List of image extensions to process
        preserve_structure: Whether to preserve subfolder structure
    """
    if extensions is None:
        extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif']
    
    # Create output folder
    Path(output_folder).mkdir(parents=True, exist_ok=True)
    
    # Find all image files
    image_files = []
    
    if preserve_structure:
        # Walk through all subdirectories
        for root, dirs, files in os.walk(input_folder):
            for file in files:
                if Path(file).suffix.lower() in extensions:
                    image_files.append(Path(root) / file)
    else:
        # Only process files in the root folder
        for file in Path(input_folder).iterdir():
            if file.is_file() and file.suffix.lower() in extensions:
                image_files.append(file)
    
    print(f"Found {len(image_files)} images to process")
    
    # Process each image with progress bar
    success_count = 0
    for img_path in tqdm(image_files, desc="Processing images"):
        if preserve_structure:
            # Preserve relative path structure
            rel_path = img_path.relative_to(input_folder)
            output_path = Path(output_folder) / rel_path
            output_path.parent.mkdir(parents=True, exist_ok=True)
        else:
            # Save all images directly in output folder
            output_path = Path(output_folder) / img_path.name
        
        if resize_and_process_image(img_path, output_path, target_size, 
                                   clip_limit, tile_grid_size):
            success_count += 1
    
    print(f"\nCompleted! Successfully processed {success_count}/{len(image_files)} images")
    print(f"Output saved to: {output_folder}")

def main():
    parser = argparse.ArgumentParser(description='Resize images to 640x640 and apply CLAHE')
    parser.add_argument('input_folder', type=str, help='Input folder containing images')
    parser.add_argument('output_folder', type=str, help='Output folder for processed images')
    parser.add_argument('--size', type=int, nargs=2, default=[640, 640],
                        help='Target size (width height), default: 640 640')
    parser.add_argument('--clip_limit', type=float, default=2.0,
                        help='CLAHE clip limit, default: 2.0')
    parser.add_argument('--tile_size', type=int, nargs=2, default=[8, 8],
                        help='CLAHE tile grid size (height width), default: 8 8')
    parser.add_argument('--preserve_structure', action='store_true',
                        help='Preserve subfolder structure in output')
    parser.add_argument('--extensions', type=str, nargs='+',
                        default=['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'],
                        help='Image extensions to process')
    
    args = parser.parse_args()
    
    # Convert tile size to tuple
    tile_grid_size = tuple(args.tile_size)
    target_size = tuple(args.size)
    
    print("=" * 50)
    print("Image Processing Pipeline")
    print("=" * 50)
    print(f"Input folder: {args.input_folder}")
    print(f"Output folder: {args.output_folder}")
    print(f"Target size: {target_size}")
    print(f"CLAHE clip limit: {args.clip_limit}")
    print(f"CLAHE tile grid size: {tile_grid_size}")
    print(f"Preserve structure: {args.preserve_structure}")
    print("=" * 50)
    
    process_folder(
        input_folder=args.input_folder,
        output_folder=args.output_folder,
        target_size=target_size,
        clip_limit=args.clip_limit,
        tile_grid_size=tile_grid_size,
        extensions=[ext.lower() for ext in args.extensions],
        preserve_structure=args.preserve_structure
    )

if __name__ == "__main__":
    main()