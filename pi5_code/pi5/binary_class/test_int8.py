# test_int8.py — run on Pi
import onnxruntime as ort
import numpy as np
import cv2
import glob

def preprocess(path):
    img = cv2.resize(cv2.imread(path), (224, 224))
    # Convert to RGB and ensure float32 explicitly
    x = np.ascontiguousarray(img[:, :, ::-1], dtype=np.float32)
    x = x.transpose(2, 0, 1)
    # Normalize - keep as float32
    x = (x / 255.0 - np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(3, 1, 1)) / \
        np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(3, 1, 1)
    return x[np.newaxis]  # Shape: (1, 3, 224, 224)

def softmax(v):
    e = np.exp(v - v.max())
    return e / e.sum()

# Load models
print("Loading FP32 model...")
fp32 = ort.InferenceSession("ncnn_export/classifier.onnx")
print("Loading INT8 model...")
int8 = ort.InferenceSession("ncnn_export/classifier_int8_v2.onnx")

fp32_name = fp32.get_inputs()[0].name
int8_name = int8.get_inputs()[0].name

# Verify input types
print(f"FP32 input type: {fp32.get_inputs()[0].type}")
print(f"INT8 input type: {int8.get_inputs()[0].type}")

print("\n" + "="*60)
for path in sorted(glob.glob("test_crops/*.jpg")):
    x = preprocess(path)
    
    # Run FP32 inference
    fp32_out = fp32.run(None, {fp32_name: x})[0][0]
    p32 = softmax(fp32_out)
    
    # Run INT8 inference - ensure same float32 input
    int8_out = int8.run(None, {int8_name: x})[0][0]
    p8 = softmax(int8_out)
    
    diff = abs(p32[1] - p8[1])
    filename = path.split('/')[-1]
    
    # Add visual indicator for significant differences
    indicator = "⚠️" if diff > 0.05 else "✅"
    
    print(f"{filename:<30} | FP32: {p32[1]:.3f} | INT8: {p8[1]:.3f} | Δ: {diff:.4f} {indicator}")

print("="*60)
