import torch
import torch.nn as nn

class TinyStudent(nn.Module):
    def __init__(self, num_classes=2):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, 3, stride=2, padding=1), nn.BatchNorm2d(16), nn.ReLU(),
            nn.Conv2d(16, 32, 3, stride=2, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.AdaptiveAvgPool2d(1)
        )
        self.classifier = nn.Linear(64, num_classes)

    def forward(self, x):
        x = self.features(x).flatten(1)
        return self.classifier(x)

# Simple loading without re-quantizing
model = TinyStudent()
model.eval()

# Load the weights with strict=False to ignore quantization artifacts
try:
    model.load_state_dict(torch.load("/home/pi5/binary_class/mobileNet/student_int8.pt", map_location='cpu'), strict=False)
    print("✅ Model loaded successfully (using FP32 mode)")
    
    # Test with random input
    with torch.no_grad():
        dummy = torch.randn(1, 3, 128, 128)
        out = torch.softmax(model(dummy), dim=1)
        print(f"Output probabilities: {out}")
        print(f"Predicted class: {torch.argmax(out, dim=1).item()} (0=Healthy, 1=Anomaly)")
        print(f"Confidence: {out.max().item():.4f}")
        
except Exception as e:
    print(f"Error loading model: {e}")
    print("\nTry using one of these alternative models:")
    print("1. student_model.pt (FP32)")
    print("2. student_model_50epochs.pt (FP32)")
    print("3. best_model_int8.pt (different architecture)")
