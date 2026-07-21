import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import os

# Paths
MODEL_PATH = "/Users/mac/Documents/PFE/rgb_binary_class/best_model_mobileNEtEnhanced.(2).pt"
TEST_DIR = "/Users/mac/Documents/PFE/test_rgb+thermal"
class_names = ["Healthy", "Anomaly"]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Load state dict
state_dict = torch.load(MODEL_PATH, map_location='cpu')

# Create MobileNetV3 Small (outputs 576 features, not 960!)
print("Creating MobileNetV3 Small backbone...")
model = models.mobilenet_v3_small(weights=None)

# The classifier from your saved model:
# Linear(576 -> 1024), then Linear(1024 -> 2)
model.classifier = nn.Sequential(
    nn.Linear(576, 1024),  # from classifier.0.weight
    nn.Hardswish(),        # MobileNetV3 activation
    nn.Dropout(0.2),
    nn.Linear(1024, 2)     # from classifier.3.weight
)

# Load the weights
print("Loading weights...")
missing, unexpected = model.load_state_dict(state_dict, strict=False)
print(f"Missing keys: {len(missing)}")
print(f"Unexpected keys: {len(unexpected)}")

# Check if we have any critical missing keys
critical_missing = [k for k in missing if 'classifier' not in k]
if critical_missing:
    print(f"\n⚠️ Warning: Missing {len(critical_missing)} backbone keys")
    print(f"   First few: {critical_missing[:3]}")
else:
    print("✅ All backbone keys matched!")

model.to(device)
model.eval()
print("\n✅ Model ready for inference!\n")

# Image preprocessing (224x224 for MobileNet)
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Get RGB images
rgb_images = [f for f in os.listdir(TEST_DIR) if f.endswith('_rgb.png') and not f.startswith('gradcam')]
rgb_images.sort()

print(f"Testing {len(rgb_images)} RGB images:")
print("=" * 60)

for img_name in rgb_images:
    img_path = os.path.join(TEST_DIR, img_name)
    
    try:
        # Load and preprocess
        image = Image.open(img_path).convert('RGB')
        input_tensor = transform(image).unsqueeze(0).to(device)
        
        # Predict
        with torch.no_grad():
            outputs = model(input_tensor)
            probabilities = torch.softmax(outputs, dim=1)
            predicted_class = torch.argmax(probabilities, dim=1).item()
            confidence = probabilities[0][predicted_class].item() * 100
        
        print(f"{img_name:20} → {class_names[predicted_class]} ({confidence:.1f}%)")
        
    except Exception as e:
        print(f"{img_name:20} → Error: {e}")

print("\n" + "=" * 60)

# Optional: Test thermal images too
print("\nTesting thermal images (first 5):")
print("-" * 50)
thermal_images = [f for f in os.listdir(TEST_DIR) if f.endswith('_thermal.png')][:5]
for img_name in thermal_images:
    img_path = os.path.join(TEST_DIR, img_name)
    
    try:
        image = Image.open(img_path).convert('RGB')
        input_tensor = transform(image).unsqueeze(0).to(device)
        
        with torch.no_grad():
            outputs = model(input_tensor)
            probabilities = torch.softmax(outputs, dim=1)
            predicted_class = torch.argmax(probabilities, dim=1).item()
            confidence = probabilities[0][predicted_class].item() * 100
        
        print(f"{img_name:20} → {class_names[predicted_class]} ({confidence:.1f}%)")
        
    except Exception as e:
        print(f"{img_name:20} → Error: {e}")