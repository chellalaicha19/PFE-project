# pi5_thermal_detector.py
# Copy to Raspberry Pi 5 with the .tflite file

import numpy as np
import cv2
import time
from pathlib import Path

# Install on Pi 5: pip install tflite-runtime opencv-python numpy
try:
    from ai_edge_litert.interpreter import Interpreter
except ImportError:
    try:
        from tflite_runtime.interpreter import Interpreter
    except ImportError:
        from tensorflow.lite.python.interpreter import Interpreter

class ThermalPanelDetector:
    def __init__(self, model_path='thermal_binary_mobilenetv2_f16.tflite', threshold=0.5):
        """Initialize the thermal anomaly detector"""
        print(f"Loading model from: {model_path}")
        self.interpreter = Interpreter(model_path=str(model_path))
        self.interpreter.allocate_tensors()
        self.threshold = threshold
        
        # Get model details
        self.input_details = self.interpreter.get_input_details()[0]
        self.output_details = self.interpreter.get_output_details()[0]
        self.input_size = 224
        
        print(f"✅ Model loaded successfully!")
        print(f"   Input shape: {self.input_details['shape']}")
        print(f"   Output shape: {self.output_details['shape']}")
        
    def preprocess(self, image):
        """
        Preprocess thermal image for model input
        Args:
            image: BGR image from OpenCV (or RGB)
        Returns:
            Preprocessed tensor (1, 224, 224, 3)
        """
        # Convert BGR to RGB if needed
        if len(image.shape) == 3 and image.shape[2] == 3:
            # Check if it's BGR (OpenCV default)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Resize to 224x224
        resized = cv2.resize(image, (self.input_size, self.input_size))
        
        # Normalize to [-1, 1] (MobileNetV2 preprocessing)
        normalized = (resized.astype(np.float32) / 127.5) - 1.0
        
        # Add batch dimension
        return np.expand_dims(normalized, axis=0)
    
    def predict(self, image):
        """
        Run inference on a single image
        Args:
            image: BGR image from OpenCV
        Returns:
            tuple: (label, confidence, raw_probability)
        """
        # Preprocess
        input_tensor = self.preprocess(image)
        
        # Set input tensor
        self.interpreter.set_tensor(self.input_details['index'], input_tensor)
        
        # Run inference
        self.interpreter.invoke()
        
        # Get output (probability of "no_anomaly" class)
        prob = float(self.interpreter.get_tensor(self.output_details['index'])[0][0])
        
        # Convert to anomaly detection output
        if prob < self.threshold:
            label = "ANOMALY"
            confidence = 1 - prob
        else:
            label = "NO_ANOMALY"
            confidence = prob
        
        return label, confidence, prob
    
    def predict_from_file(self, image_path):
        """Load image from file and predict"""
        image = cv2.imread(str(image_path))
        if image is None:
            raise ValueError(f"Could not read image: {image_path}")
        return self.predict(image)
    
    def predict_batch(self, images):
        """Run inference on a batch of images"""
        results = []
        for img in images:
            results.append(self.predict(img))
        return results

def main():
    """Test the detector with sample images"""
    # Initialize detector
    detector = ThermalPanelDetector(threshold=0.5)
    
    # Test on images in 'test_images' folder
    test_dir = Path('test_images')
    if test_dir.exists():
        image_files = list(test_dir.glob('*.jpg')) + list(test_dir.glob('*.png'))
        
        if image_files:
            print(f"\n📸 Testing {len(image_files)} images...")
            print("=" * 60)
            
            for img_path in image_files:
                # Predict
                start_time = time.time()
                label, confidence, prob = detector.predict_from_file(img_path)
                inference_time = (time.time() - start_time) * 1000
                
                # Display results
                print(f"\n📷 {img_path.name}")
                print(f"   Prediction: {label}")
                print(f"   Confidence: {confidence:.1%}")
                print(f"   Raw score: {prob:.4f}")
                print(f"   Inference: {inference_time:.1f} ms")
        else:
            print("\n⚠️ No images found in 'test_images' folder")
    else:
        print("\n📁 Create a 'test_images' folder and add thermal images")
        print("   mkdir test_images")
        print("   cp /path/to/thermal/images/*.jpg test_images/")
    
    # Performance benchmark
    print("\n" + "=" * 60)
    print("🚀 Performance Benchmark")
    print("=" * 60)
    
    # Create dummy image for benchmarking
    dummy_image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    
    # Warm-up
    for _ in range(10):
        detector.predict(dummy_image)
    
    # Benchmark
    times = []
    for _ in range(100):
        start = time.time()
        detector.predict(dummy_image)
        times.append((time.time() - start) * 1000)
    
    print(f"Average inference time: {np.mean(times):.1f} ms")
    print(f"Min inference time: {np.min(times):.1f} ms")
    print(f"Max inference time: {np.max(times):.1f} ms")
    print(f"FPS: {1000/np.mean(times):.1f} fps")

if __name__ == "__main__":
    main()