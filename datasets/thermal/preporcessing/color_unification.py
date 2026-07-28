import cv2
import numpy as np
import os
from pathlib import Path

def unify_thermal_color(image_path, output_path, color_map='INFERNO'):
    """
    Convert thermal images to consistent color representation.
    Hot = orange/yellow, Cold = purple/blue
    """
    img = cv2.imread(image_path)
    
    # If image is already pseudocolor, convert to grayscale first
    if len(img.shape) == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img
    
    # Apply consistent colormap
    if color_map == 'INFERNO':
        # Orange/yellow for hot, dark purple/black for cold
        colored = cv2.applyColorMap(gray, cv2.COLORMAP_INFERNO)
    elif color_map == 'JET':
        # Alternative: blue (cold) -> cyan -> yellow -> red (hot)
        colored = cv2.applyColorMap(gray, cv2.COLORMAP_JET)
    
    # Optional: Normalize to [0,255] range
    colored = cv2.normalize(colored, None, 0, 255, cv2.NORM_MINMAX)
    
    cv2.imwrite(output_path, colored)

# Apply to your entire dataset
for split in ['train', 'val', 'test']:
    for category in ['full_cold', 'partial_cold', 'hotspot', 'no_anomaly']:
        input_dir = f'thermal_unified/{split}/{category}'
        output_dir = f'thermal_unified_processed/{split}/{category}'
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        for img_file in os.listdir(input_dir):
            if img_file.endswith(('.jpg', '.png', '.jpeg')):
                unify_thermal_color(
                    f'{input_dir}/{img_file}',
                    f'{output_dir}/{img_file}'
                )