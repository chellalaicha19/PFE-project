# Add at the end of your extract_rgb_embedding.py or create evaluate_rgb_test.py
import torch
from torchvision import transforms
from pathlib import Path
from PIL import Image
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# Load your model (same as in your script)
model = torch.load("/Users/mac/Documents/PFE/rgb_multicalssification/best_rgb_multiclass_final.pt", map_location='cpu')
model.eval()

class_names = ["Soiling_Pollution", "Shadowing_Vegetation", "Burn_Discoloration", "Structural_Damage"]  # add Clean if you have 5 classes

test_dir = Path("/Users/mac/Documents/PFE/rgb_multicalssification/test")  # or wherever your test subfolders are

y_true, y_pred = [], []

transform = transforms.Compose([  # must match your training preprocessing
    transforms.Resize((224, 224)),  # or whatever size you used
    transforms.ToTensor(),
    # add your exact normalization / CLAHE if you have it
])

for class_dir in test_dir.iterdir():
    if not class_dir.is_dir(): continue
    true_class = class_names.index(class_dir.name) if class_dir.name in class_names else -1
    for img_path in class_dir.glob("*.jpg"):
        img = Image.open(img_path).convert("RGB")
        tensor = transform(img).unsqueeze(0)
        with torch.no_grad():
            logits = model(tensor)
            pred = logits.argmax(dim=1).item()
        y_true.append(true_class)
        y_pred.append(pred)

print(classification_report(y_true, y_pred, target_names=class_names))
cm = confusion_matrix(y_true, y_pred)
sns.heatmap(cm, annot=True, fmt='d', xticklabels=class_names, yticklabels=class_names)
plt.title("RGB Classifier Confusion Matrix")
plt.show()