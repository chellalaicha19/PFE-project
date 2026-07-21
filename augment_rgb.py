import os
import cv2
import albumentations as A
from tqdm import tqdm
import random

# ====================== CONFIG ======================
BASE_DIR = "unified_dataset"
OUTPUT_DIR = "prepared/unified_dataset_augmented"
IMAGE_SIZE = 224   # All output images will be this size

TARGET_COUNTS = {
    0: 2500,   # Clean
    1: 2500,   # Soiling
    2: 2200,   # Shadowing
    3: 2200,   # Burn/Discoloration
    4: 2200    # Structural Damage
}

AUGMENTATION_PIPELINES = {
    0: A.Compose([                          # Light for Clean
        A.RandomRotate90(p=0.5),
        A.HorizontalFlip(p=0.5),
        A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.4),
        A.GaussNoise(var_limit=(10, 50), p=0.3),
    ]),

    1: A.Compose([                          # Medium for Soiling
        A.HorizontalFlip(p=0.7),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.6),
        A.RandomBrightnessContrast(brightness_limit=0.25, contrast_limit=0.25, p=0.7),
        A.GaussNoise(var_limit=(10, 50), p=0.4),
    ]),

    2: A.Compose([                          # Strong for Shadowing
        A.HorizontalFlip(p=0.8),
        A.VerticalFlip(p=0.6),
        A.RandomRotate90(p=0.8),
        A.RandomBrightnessContrast(brightness_limit=0.35, contrast_limit=0.35, p=0.8),
        A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.15, rotate_limit=30, p=0.7),
        A.GaussNoise(var_limit=(10, 50), p=0.5),
    ]),

    3: A.Compose([                          # Burn/Discoloration
        A.HorizontalFlip(p=0.9),
        A.VerticalFlip(p=0.7),
        A.RandomRotate90(p=0.9),
        # Capped at 0.25 — burns are defined by color signature,
        # aggressive brightness shifts can erase the fault appearance
        A.RandomBrightnessContrast(brightness_limit=0.25, contrast_limit=0.25, p=0.7),
        A.ShiftScaleRotate(shift_limit=0.12, scale_limit=0.2, rotate_limit=35, p=0.8),
        A.GaussNoise(var_limit=(20, 60), p=0.6),
        A.CLAHE(clip_limit=2.0, p=0.4),    # Helps highlight burns, keep clip_limit moderate
    ]),

    4: A.Compose([                          # Strong for Structural Damage
        A.HorizontalFlip(p=0.8),
        A.VerticalFlip(p=0.6),
        A.RandomRotate90(p=0.8),
        A.RandomBrightnessContrast(brightness_limit=0.25, contrast_limit=0.25, p=0.7),
        A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.15, rotate_limit=30, p=0.7),
        A.GaussNoise(var_limit=(10, 50), p=0.5),
    ])
}

# ====================================================

os.makedirs(OUTPUT_DIR, exist_ok=True)

for class_name in sorted(os.listdir(BASE_DIR)):
    if not class_name.startswith("Class_"):
        continue

    class_idx = int(class_name.split("_")[1])
    input_folder = os.path.join(BASE_DIR, class_name)
    output_folder = os.path.join(OUTPUT_DIR, class_name)
    os.makedirs(output_folder, exist_ok=True)

    images = [f for f in os.listdir(input_folder)
              if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    original_count = len(images)
    target = TARGET_COUNTS.get(class_idx, original_count)

    print(f"\nProcessing {class_name} → {original_count} originals → target {target}")

    # Copy originals first (resize to standard size)
    copied = 0
    for img_name in tqdm(images, desc=f"Copying originals"):
        src = os.path.join(input_folder, img_name)
        dst = os.path.join(output_folder, img_name)
        if not os.path.exists(dst):
            img = cv2.imread(src)
            if img is None:
                print(f"  [WARN] Could not read {src}, skipping.")
                continue
            img = cv2.resize(img, (IMAGE_SIZE, IMAGE_SIZE), interpolation=cv2.INTER_LINEAR)
            cv2.imwrite(dst, img)
            copied += 1

    # Generate augmented images until target
    current_count = original_count
    transform = AUGMENTATION_PIPELINES[class_idx]

    idx = 0
    pbar = tqdm(total=target - current_count, desc=f"Augmenting")
    while current_count < target:
        img_name = random.choice(images)
        img_bgr = cv2.imread(os.path.join(input_folder, img_name))
        if img_bgr is None:
            continue

        # Resize source before augmenting
        img_bgr = cv2.resize(img_bgr, (IMAGE_SIZE, IMAGE_SIZE), interpolation=cv2.INTER_LINEAR)

        # BGR → RGB (Albumentations expects RGB)
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

        augmented_rgb = transform(image=img_rgb)['image']

        # RGB → BGR for saving with OpenCV
        augmented_bgr = cv2.cvtColor(augmented_rgb, cv2.COLOR_RGB2BGR)

        new_name = f"aug_{idx:05d}_{img_name}"
        cv2.imwrite(os.path.join(output_folder, new_name), augmented_bgr)

        current_count += 1
        idx += 1
        pbar.update(1)

    pbar.close()
    print(f"  Done: {current_count} images in {output_folder}")

print("\nAugmentation complete. Dataset ready at:", OUTPUT_DIR)