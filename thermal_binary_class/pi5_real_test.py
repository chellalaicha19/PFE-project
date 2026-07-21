# pi5_thermal_anomaly_monitor.py
# Copy to Raspberry Pi 5 with the .tflite file

import numpy as np
import cv2
import time
import csv
import threading
import queue
import signal
import sys
import os
from pathlib import Path
from datetime import datetime
from collections import deque

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

class ThermalAnomalyMonitor:
    def __init__(self, model_path='thermal_binary_mobilenetv2_f16.tflite', 
                 threshold=0.5, capture_interval_sec=0.125):  # 0.125 sec = 8 FPS
        """Initialize the thermal anomaly monitoring system"""
        
        # Initialize detector
        self.detector = ThermalPanelDetector(model_path, threshold)
        
        # Configuration
        self.capture_interval = capture_interval_sec
        self.target_captures = 10
        self.captures_done = 0
        self.anomalies_saved = 0
        
        # Storage paths
        self.base_dir = Path(__file__).parent
        self.anomaly_dir = self.base_dir / 'anomalies'
        self.logs_dir = self.base_dir / 'logs'
        
        # Create directories
        self.anomaly_dir.mkdir(exist_ok=True)
        self.logs_dir.mkdir(exist_ok=True)
        
        # Setup CSV log file
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.csv_path = self.logs_dir / f'anomaly_detection_log_{timestamp}.csv'
        self.setup_csv_log()
        
        # Control flags
        self.running = True
        self.capture_queue = queue.Queue(maxsize=20)
        
        # Statistics
        self.frame_count = 0
        self.start_time = None
        
        print(f"\n📁 Storage paths:")
        print(f"   Anomalies: {self.anomaly_dir}")
        print(f"   Log file: {self.csv_path}")
        print(f"\n⚙️ Configuration:")
        print(f"   Capture rate: {1/self.capture_interval} FPS")
        print(f"   Target captures: {self.target_captures}")
        print(f"   Detection threshold: {threshold}")
        
    def setup_csv_log(self):
        """Initialize CSV log file with headers"""
        with open(self.csv_path, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow([
                'timestamp', 
                'capture_number', 
                'classification', 
                'confidence', 
                'raw_score',
                'image_saved',
                'image_path'
            ])
        print(f"✅ CSV log created: {self.csv_path}")
    
    def log_to_csv(self, capture_num, label, confidence, raw_score, image_saved, image_path):
        """Log detection result to CSV file"""
        with open(self.csv_path, 'a', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow([
                datetime.now().isoformat(),
                capture_num,
                label,
                f"{confidence:.4f}",
                f"{raw_score:.4f}",
                'Yes' if image_saved else 'No',
                image_path if image_saved else ''
            ])
    
    def save_anomaly_image(self, image, capture_num, label, confidence):
        """Save image as anomaly with metadata filename"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]  # microsecond precision
        # Create descriptive filename: anomaly_capture#_timestamp_conf%.2f.jpg
        filename = f"anomaly_cap{capture_num:03d}_{timestamp}_conf{confidence:.2f}.jpg"
        filepath = self.anomaly_dir / filename
        
        # Save colored thermal image
        cv2.imwrite(str(filepath), image)
        
        # Also save raw grayscale version for analysis
        gray_filename = f"anomaly_cap{capture_num:03d}_{timestamp}_gray.jpg"
        gray_path = self.anomaly_dir / gray_filename
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        cv2.imwrite(str(gray_path), gray_image)
        
        return str(filepath)
    
    def process_frame(self, frame, capture_num):
        """Process single frame: predict, log, save if anomaly"""
        
        # Convert frame (already BGR from camera)
        # For thermal camera, frame is already processed to BGR with colormap
        
        # Run prediction
        start_time = time.time()
        label, confidence, raw_score = self.detector.predict(frame)
        inference_time = (time.time() - start_time) * 1000
        
        # Check if anomaly
        is_anomaly = (label == "ANOMALY")
        saved_path = ""
        
        if is_anomaly:
            # Save the colored thermal image
            saved_path = self.save_anomaly_image(frame, capture_num, label, confidence)
            self.anomalies_saved += 1
            print(f"⚠️  ANOMALY DETECTED! (Capture #{capture_num})")
            print(f"   Confidence: {confidence:.1%} | Saved: {saved_path}")
        else:
            print(f"✓ Normal (Capture #{capture_num}) | Confidence: {confidence:.1%}")
        
        # Log to CSV
        self.log_to_csv(capture_num, label, confidence, raw_score, is_anomaly, saved_path)
        
        # Print progress
        print(f"   Inference: {inference_time:.1f}ms | Progress: {capture_num}/{self.target_captures}")
        print("-" * 60)
        
        return is_anomaly
    
    def capture_and_classify(self):
        """Main capture loop with classification"""
        
        # Initialize camera
        print("\n📷 Initializing TC002C Duo thermal camera...")
        cap = cv2.VideoCapture(0, cv2.CAP_V4L2)
        cap.set(cv2.CAP_PROP_CONVERT_RGB, 0)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 256)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 192)
        
        # Verify camera settings
        actual_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        print(f"✅ Camera initialized: {actual_width}x{actual_height}")
        
        self.start_time = time.time()
        capture_count = 0
        
        print(f"\n🎯 Starting monitoring - Target: {self.target_captures} captures at {1/self.capture_interval:.1f} FPS")
        print("=" * 60)
        
        # Signal handler for graceful shutdown
        def signal_handler(sig, frame):
            print("\n\n🛑 Received interrupt signal. Shutting down gracefully...")
            self.running = False
        
        signal.signal(signal.SIGINT, signal_handler)
        
        # Main capture loop
        last_capture_time = 0
        
        while self.running and capture_count < self.target_captures:
            # Read frame from camera
            ret, frame = cap.read()
            if not ret:
                print("❌ Failed to read frame from camera")
                time.sleep(0.001)
                continue
            
            # Process thermal frame (same as original script)
            # Convert YUV to BGR
            bgr = cv2.cvtColor(frame, cv2.COLOR_YUV2BGR_YUYV)
            
            # Apply inferno colormap for better visualization
            gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
            colored = cv2.applyColorMap(gray, cv2.COLORMAP_INFERNO)
            
            # Rate limiting for captures
            current_time = time.time()
            if current_time - last_capture_time >= self.capture_interval:
                capture_count += 1
                
                # Process the frame for classification
                self.process_frame(colored, capture_count)
                
                last_capture_time = current_time
                self.frame_count += 1
            
            # Small delay to prevent CPU overload
            time.sleep(0.001)
        
        # Cleanup
        cap.release()
        
        # Print summary
        self.print_summary()
    
    def print_summary(self):
        """Print monitoring summary"""
        elapsed_time = time.time() - self.start_time if self.start_time else 0
        
        print("\n" + "=" * 60)
        print("📊 MONITORING SUMMARY")
        print("=" * 60)
        print(f"✅ Target captures completed: {self.target_captures}/{self.target_captures}")
        print(f"⚠️  Anomalies detected: {self.anomalies_saved}")
        print(f"📁 Anomaly images saved to: {self.anomaly_dir}")
        print(f"📄 Detection log saved to: {self.csv_path}")
        print(f"⏱️  Total time: {elapsed_time:.1f} seconds")
        print(f"📸 Effective capture rate: {self.target_captures/elapsed_time:.1f} FPS")
        
        if self.anomalies_saved > 0:
            print(f"\n🔍 Anomaly rate: {self.anomalies_saved/self.target_captures*100:.1f}%")
        else:
            print("\n✨ No anomalies detected in this session")
        
        print("\n🏁 Monitoring completed successfully!")

def main():
    """Main entry point"""
    print("=" * 60)
    print("🔥 Thermal Anomaly Monitor for Raspberry Pi 5")
    print("=" * 60)
    
    # Configuration
    MODEL_PATH = 'thermal_binary_mobilenetv2_f16.tflite'
    THRESHOLD = 0.5  # Adjust based on your needs (0.3-0.7 typically)
    CAPTURE_FPS = 8  # 8 frames per second
    CAPTURE_INTERVAL = 1.0 / CAPTURE_FPS
    
    # Check if model exists
    if not Path(MODEL_PATH).exists():
        print(f"❌ Model file not found: {MODEL_PATH}")
        print("Please ensure the .tflite model file is in the same directory")
        return
    
    # Create and run monitor
    try:
        monitor = ThermalAnomalyMonitor(
            model_path=MODEL_PATH,
            threshold=THRESHOLD,
            capture_interval_sec=CAPTURE_INTERVAL
        )
        
        monitor.capture_and_classify()
        
    except KeyboardInterrupt:
        print("\n\n🛑 Stopped by user")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()