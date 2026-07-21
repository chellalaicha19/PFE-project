# quantize_fp16.py — run on Mac
import onnx
from onnxconverter_common import float16
import onnxruntime as ort
import numpy as np
import cv2
import glob

# Convert to FP16
print("Converting model to FP16...")
model = onnx.load("/Users/mac/Documents/PFE/rgb_binary_class/onnx/classifier.onnx")
model_fp16 = float16.convert_float_to_float16(model, keep_io_types=True)
onnx.save(model_fp16, "classifier_fp16.onnx")
print("Saved: classifier_fp16.onnx")

# Quick sanity check on Mac
def softmax(x):
    e = np.exp(x - x.max())
    return e / e.sum()

MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(3, 1, 1)
STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(3, 1, 1)

print("\nLoading models...")
fp32 = ort.InferenceSession("/Users/mac/Documents/PFE/rgb_binary_class/onnx/classifier.onnx")
fp16 = ort.InferenceSession("classifier_fp16.onnx")
name = fp32.get_inputs()[0].name

print(f"FP32 input type: {fp32.get_inputs()[0].type}")
print(f"FP16 input type: {fp16.get_inputs()[0].type}")

# Test on a few calibration crops
test_imgs = glob.glob("/Users/mac/Documents/test_crops/*.jpg",
                      recursive=True)[:20]

print(f"\nTesting on {len(test_imgs)} images...")
print("=" * 80)

deltas = []
for path in test_imgs:
    img = cv2.resize(cv2.imread(path), (224, 224))
    x = np.ascontiguousarray(img[:, :, ::-1], dtype=np.float32).transpose(2, 0, 1)
    x = (x / 255.0 - MEAN) / STD
    x = x[np.newaxis]
    
    p32 = softmax(fp32.run(None, {name: x})[0][0])
    p16 = softmax(fp16.run(None, {name: x.astype(np.float32)})[0][0])  # Keep as float32 input
    d = abs(p32[1] - p16[1])
    deltas.append(d)
    status = "✅" if d < 0.01 else ("🟡" if d < 0.05 else "⚠️")
    print(f"{path.split('/')[-1]:<40} FP32:{p32[1]:.4f}  FP16:{p16[1]:.4f}  Δ:{d:.5f} {status}")

print("=" * 80)
print(f"\nMean Δ: {np.mean(deltas):.6f}")
print(f"Max Δ:  {np.max(deltas):.6f}")
print(f"Std Δ:  {np.std(deltas):.6f}")

if np.max(deltas) < 0.01:
    print("\n✅ PERFECT! FP16 is safe to deploy (virtually lossless)")
elif np.max(deltas) < 0.05:
    print("\n✅ GOOD! FP16 is safe to deploy (minimal loss)")
else:
    print("\n⚠️ WARNING: Some differences detected, but likely still acceptable")

print("\n📁 Files created:")
print("   - classifier_fp16.onnx (deploy this to your Pi)")