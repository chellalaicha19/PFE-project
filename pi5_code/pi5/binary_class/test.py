import onnxruntime as ort
import numpy as np
import time
from PIL import Image
import torchvision.transforms as transforms
import cv2  # Alternative if you prefer OpenCV

def preprocess_image(image_path, target_size=(224, 224)):
    """
    Preprocess image for the model
    Using same transforms as during training
    """
    # Method 1: Using PIL + torchvision (recommended)
    image = Image.open(image_path).convert('RGB')
    
    # Define preprocessing transforms (adjust based on your training)
    transform = transforms.Compose([
        transforms.Resize(target_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],  # ImageNet stats
                           std=[0.229, 0.224, 0.225])
    ])
    
    img_tensor = transform(image)
    img_numpy = img_tensor.numpy()
    img_numpy = img_numpy.reshape(1, 3, target_size[0], target_size[1])
    
    return img_numpy.astype(np.float32)

def preprocess_image_opencv(image_path, target_size=(224, 224)):
    """
    Alternative preprocessing using OpenCV (faster)
    """
    # Read image with OpenCV (BGR format)
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, target_size)
    
    # Convert to float and normalize (0-1 range)
    img = img.astype(np.float32) / 255.0
    
    # Normalize using ImageNet stats
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    img = (img - mean) / std
    
    # Change to NCHW format (batch, channels, height, width)
    img = img.transpose(2, 0, 1)
    img = img.reshape(1, 3, target_size[0], target_size[1])
    
    return img.astype(np.float32)

# Load model
model_path = "ncnn_export/classifier.onnx"  # Update with your actual path
sess = ort.InferenceSession(model_path)

# Load and preprocess your image
image_path = input("Enter image path: ")  # Or specify directly: "path/to/your/image.jpg"
# Example: image_path = "test_image.jpg"

try:
    # Use PIL method (choose one)
    input_data = preprocess_image(image_path)
    
    # Or use OpenCV method (uncomment if you prefer)
    # input_data = preprocess_image_opencv(image_path)
    
    print(f"Input shape: {input_data.shape}")
    print(f"Input range: [{input_data.min():.3f}, {input_data.max():.3f}]")
    
    # Warmup
    dummy = np.random.randn(1, 3, 224, 224).astype(np.float32)
    sess.run(None, {"input": dummy})
    
    # Single inference with timing
    start = time.perf_counter()
    output = sess.run(None, {"input": input_data})[0]
    inference_time = (time.perf_counter() - start) * 1000
    
    # Benchmark multiple inferences (optional)
    n_runs = 50
    times = []
    for _ in range(n_runs):
        t = time.perf_counter()
        sess.run(None, {"input": input_data})
        times.append((time.perf_counter() - t) * 1000)
    
    # Process output (assuming binary classification)
    probability = 1 / (1 + np.exp(-output[0]))  # Sigmoid for single output
    prediction = 1 if probability > 0.5 else 0
    
    print(f"\n{'='*50}")
    print(f"Image: {image_path}")
    print(f"Raw output: {output[0]:.6f}")
    print(f"Probability (class 1): {probability:.4f}")
    print(f"Prediction: {'Class 1' if prediction == 1 else 'Class 0'}")
    print(f"\nSingle inference time: {inference_time:.2f} ms")
    print(f"Average over {n_runs} runs: {sum(times)/len(times):.2f} ms")
    print(f"Min: {min(times):.2f} ms, Max: {max(times):.2f} ms")
    print(f"{'='*50}")
    
except FileNotFoundError:
    print(f"Error: Image not found at {image_path}")
except Exception as e:
    print(f"Error: {e}")
