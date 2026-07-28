import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import sys
import time  # Line 1: Import time module

CLASS_NAMES = ["Healthy", "Anomaly"]
IMG_SIZE = 224

def load_model(weights_path):
    model = models.mobilenet_v3_small(weights=None)
    model.classifier[3] = nn.Linear(model.classifier[3].in_features, 2)
    model.load_state_dict(torch.load(weights_path, map_location="cpu"))
    model.eval()
    return model

def predict(model, image_path):
    tf = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225]),
    ])
    img = Image.open(image_path).convert("RGB")
    tensor = tf(img).unsqueeze(0)          # [1, 3, 224, 224]
    
    start_time = time.time()  # Line 2: Start timing
    with torch.no_grad():
        logits = model(tensor)
        probs  = torch.softmax(logits, dim=1)[0]
        pred   = probs.argmax().item()
    end_time = time.time()    # Line 3: End timing
    
    print(f"Prediction : {CLASS_NAMES[pred]}")
    print(f"Confidence : Healthy={probs[0]:.3f}  Anomaly={probs[1]:.3f}")
    print(f"Classification time: {(end_time - start_time)*1000:.2f} ms")  # Line 4: Display timing

if __name__ == "__main__":
    model = load_model("best_model_mobileNEtEnhanced.pt")
    predict(model, sys.argv[1])   # pass image path as argument
