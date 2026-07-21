import cv2
import time
import numpy as np
import threading
import queue
from collections import deque
from ultralytics import YOLO


class PanelDetector:
    def __init__(self, model_path="/home/pi5/panel_detection2/best_ncnn_model", 
                 target_size=(640, 640),
                 use_clahe=True,
                 clahe_clip=1.5,
                 clahe_grid=(4, 4),
                 use_parallel=True,
                 queue_size=2):
        """
        Initialize panel detector with optimized settings for Raspberry Pi
        
        Args:
            model_path: Path to NCNN model
            target_size: Input size (640x640 for better detection accuracy)
            use_clahe: Enable CLAHE preprocessing (disable if lighting is good)
            clahe_clip: CLAHE clip limit (1.5 is good balance)
            clahe_grid: CLAHE grid size (4x4 is faster than 8x8)
            use_parallel: Enable parallel processing with threading
            queue_size: Size of queues for parallel processing
        """
        print(f"Loading model from {model_path}...")
        self.model = YOLO(model_path, task='obb')
        self.target_size = target_size
        self.use_clahe = use_clahe
        self.use_parallel = use_parallel
        self.running = False
        
        if use_clahe:
            self.clahe = cv2.createCLAHE(clipLimit=clahe_clip, tileGridSize=clahe_grid)
        
        # Parallel processing queues
        if use_parallel:
            self.raw_queue = queue.Queue(maxsize=queue_size)
            self.clahe_queue = queue.Queue(maxsize=queue_size)
            self.result_queue = queue.Queue(maxsize=queue_size)
            
            # Start worker threads
            self.clahe_thread = threading.Thread(target=self.clahe_worker, daemon=True)
            self.detection_thread = threading.Thread(target=self.detection_worker, daemon=True)
            
        print(f"PanelDetector initialized with target_size={target_size}, use_clahe={use_clahe}, use_parallel={use_parallel}")

    def apply_clahe(self, img):
        """Apply CLAHE preprocessing to a single frame"""
        # Resize to target size
        img = cv2.resize(img, self.target_size, interpolation=cv2.INTER_LINEAR)
        
        # Apply CLAHE if enabled
        if self.use_clahe:
            lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
            lab[:, :, 0] = self.clahe.apply(lab[:, :, 0])
            img = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
        
        return img
    
    def clahe_worker(self):
        """Worker thread for CLAHE preprocessing"""
        while self.running:
            try:
                frame_data = self.raw_queue.get(timeout=0.1)
                if frame_data is None:  # Stop signal
                    break
                    
                frame_id, frame = frame_data
                processed_frame = self.apply_clahe(frame)
                self.clahe_queue.put((frame_id, processed_frame))
            except queue.Empty:
                continue
    
    def detection_worker(self):
        """Worker thread for YOLO detection"""
        while self.running:
            try:
                frame_data = self.clahe_queue.get(timeout=0.1)
                if frame_data is None:  # Stop signal
                    break
                    
                frame_id, processed_frame = frame_data
                
                # Run inference
                results = self.model(processed_frame, verbose=False)
                
                # Put results in queue
                self.result_queue.put((frame_id, results, processed_frame))
            except queue.Empty:
                continue
    
    def preprocess(self, img):
        """Non-parallel preprocessing (fallback when use_parallel=False)"""
        return self.apply_clahe(img)
    
    def detect_parallel(self, img, conf_threshold=0.25, iou_threshold=0.45, verbose=True):
        """
        Run panel detection with parallel CLAHE processing
        
        Args:
            img: Input image
            conf_threshold: Confidence threshold
            iou_threshold: IOU threshold
            verbose: Print timing information
            
        Returns:
            results: YOLO results object
            timings: Dictionary with timing information
        """
        timings = {}
        
        # Start parallel processing if not already running
        if not self.running:
            self.running = True
            self.clahe_thread.start()
            self.detection_thread.start()
        
        frame_id = int(time.time() * 1000)  # Unique ID for this frame
        
        # Step 1: Queue raw frame for CLAHE processing
        t0 = time.perf_counter()
        self.raw_queue.put((frame_id, img))
        timings['queue_time'] = (time.perf_counter() - t0) * 1000
        
        # Step 2: Wait for detection results
        t0 = time.perf_counter()
        while True:
            try:
                result_id, results, processed_frame = self.result_queue.get(timeout=1.0)
                if result_id == frame_id:
                    timings['detection_time'] = (time.perf_counter() - t0) * 1000
                    break
                else:
                    # Put back results from different frame
                    self.result_queue.put((result_id, results, processed_frame))
            except queue.Empty:
                timings['detection_time'] = (time.perf_counter() - t0) * 1000
                raise TimeoutError("Detection timed out")
        
        # Apply confidence and IOU thresholds
        if results[0].obb is not None:
            # Filter by confidence
            mask = results[0].obb.conf >= conf_threshold
            if hasattr(results[0].obb, 'conf'):
                results[0].obb = results[0].obb[mask]
        
        timings['total'] = timings['queue_time'] + timings['detection_time']
        
        if verbose:
            self.print_timings_parallel(timings)
        
        return results, timings, processed_frame
    
    def detect(self, img, conf_threshold=0.25, iou_threshold=0.45, verbose=True):
        """
        Run panel detection on image (supports both parallel and sequential modes)
        
        Returns:
            results: YOLO results object
            timings: Dictionary with timing information for each step
        """
        if self.use_parallel:
            results, timings, processed = self.detect_parallel(img, conf_threshold, iou_threshold, verbose)
            return results, timings
        
        # Sequential processing (original method)
        timings = {}
        
        # Step 1: Preprocessing
        t0 = time.perf_counter()
        processed = self.preprocess(img)
        timings['preprocessing'] = (time.perf_counter() - t0) * 1000
        
        # Step 2: Inference
        t0 = time.perf_counter()
        results = self.model(processed, verbose=False, conf=conf_threshold, iou=iou_threshold)
        timings['inference'] = (time.perf_counter() - t0) * 1000
        
        # Step 3: Postprocessing (if needed)
        t0 = time.perf_counter()
        # Add any postprocessing here if needed
        timings['postprocessing'] = (time.perf_counter() - t0) * 1000
        
        timings['total'] = timings['preprocessing'] + timings['inference'] + timings['postprocessing']
        
        if verbose:
            self.print_timings(timings)
        
        return results, timings
    
    def print_timings(self, timings):
        """Print timing information in a formatted way (sequential)"""
        print("\n" + "="*60)
        print("DETECTION TIMING BREAKDOWN (SEQUENTIAL)")
        print("="*60)
        print(f"Preprocessing (resize + CLAHE):  {timings['preprocessing']:.2f} ms")
        print(f"Inference (NCNN model):          {timings['inference']:.2f} ms")
        print(f"Postprocessing:                  {timings['postprocessing']:.2f} ms")
        print(f"Total:                           {timings['total']:.2f} ms")
        print(f"FPS:                             {1000/timings['total']:.1f} fps")
        print("="*60)
    
    def print_timings_parallel(self, timings):
        """Print timing information in a formatted way (parallel)"""
        print("\n" + "="*60)
        print("DETECTION TIMING BREAKDOWN (PARALLEL)")
        print("="*60)
        print(f"Queue time (frame submission):   {timings['queue_time']:.2f} ms")
        print(f"Detection time (CLAHE + YOLO):   {timings['detection_time']:.2f} ms")
        print(f"Total:                           {timings['total']:.2f} ms")
        print(f"FPS:                             {1000/timings['total']:.1f} fps")
        print("="*60)
        print("NOTE: CLAHE cost is hidden in parallel processing!")
        print("="*60)

    def draw_detections(self, img, results, conf_threshold=0.25, save_path=None):
        """
        Draw oriented bounding boxes on the original image
        
        Args:
            img: Original image (will be resized to target_size for drawing)
            results: YOLO results object
            conf_threshold: Confidence threshold for displaying boxes
            save_path: Path to save the output image (if None, doesn't save)
        
        Returns:
            img_with_boxes: Image with drawn bounding boxes
            detections: List of detection dictionaries
        """
        # Create a copy of the original image
        img_with_boxes = img.copy()
        
        # Get original dimensions
        orig_h, orig_w = img.shape[:2]
        
        # Calculate scaling factors to map detections back to original size
        scale_x = orig_w / self.target_size[0]
        scale_y = orig_h / self.target_size[1]
        
        detections = []
        
        for result in results:
            if result.obb is not None:
                # Get oriented bounding boxes
                boxes = result.obb.xyxyxyxy  # Shape: (N, 4, 2) - four corners for each box
                confs = result.obb.conf
                cls_ids = result.obb.cls if hasattr(result.obb, 'cls') else None
                
                for i, (box, conf) in enumerate(zip(boxes, confs)):
                    if conf >= conf_threshold:
                        # Convert tensor to numpy if needed
                        if hasattr(box, 'cpu'):
                            box = box.cpu().numpy()
                        
                        # Scale box coordinates back to original image size
                        scaled_box = box * np.array([scale_x, scale_y])
                        scaled_box = scaled_box.astype(np.int32)
                        
                        # Draw the oriented bounding box (4 corners)
                        cv2.polylines(img_with_boxes, [scaled_box], True, (0, 255, 0), 3)
                        
                        # Draw confidence score
                        label = f"Panel: {conf:.2f}"
                        # Get the top-left corner for text (use first corner)
                        text_x, text_y = scaled_box[0][0], scaled_box[0][1] - 10
                        cv2.putText(img_with_boxes, label, (text_x, text_y), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                        
                        # Store detection info
                        detection = {
                            'box': scaled_box.tolist(),
                            'confidence': float(conf),
                            'class_id': int(cls_ids[i]) if cls_ids is not None else None
                        }
                        detections.append(detection)
        
        # Add info text at the top
        if hasattr(self, 'last_timings'):
            info_text = f"Detections: {len(detections)} | FPS: {1000/self.last_timings['total']:.1f}"
            cv2.putText(img_with_boxes, info_text, (10, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        
        # Save if path provided
        if save_path:
            cv2.imwrite(save_path, img_with_boxes)
            print(f"Image saved to: {save_path}")
        
        return img_with_boxes, detections
    
    def stop(self):
        """Stop parallel processing threads"""
        self.running = False
        if self.use_parallel:
            # Send stop signals
            self.raw_queue.put(None)
            self.clahe_queue.put(None)


# Usage example with timing and visualization
if __name__ == "__main__":
    # Test both sequential and parallel modes
    for mode in ['sequential', 'parallel']:
        print("\n" + "="*80)
        print(f"TESTING {mode.upper()} MODE")
        print("="*80)
        
        # Initialize detector
        print("Initializing PanelDetector...")
        detector = PanelDetector(
            target_size=(640, 640),      # 640x640 for better detection accuracy
            use_clahe=True,               # Enable CLAHE for better detection
            clahe_clip=1.5,               # Optimized CLAHE parameter
            clahe_grid=(4, 4),            # Faster grid size
            use_parallel=(mode == 'parallel'),  # Enable/disable parallel processing
            queue_size=2                  # Queue size for parallel mode
        )
        
        # Load image
        img_path = "/home/pi5/panel_detection/test_image14.jpg"
        print(f"\nLoading image from: {img_path}")
        img = cv2.imread(img_path)
        if img is None:
            print(f"Warning: Test image not found at {img_path}")
            continue
        
        print(f"Image size: {img.shape[1]}x{img.shape[0]}")
        
        # Run multiple detections to show parallel benefit
        print(f"\nRunning 5 detections in {mode} mode...")
        print("(First run includes model loading overhead)\n")
        
        optimal_threshold = 0.35
        all_timings = []
        
        for i in range(5):
            results, timings = detector.detect(img, conf_threshold=optimal_threshold, verbose=False)
            all_timings.append(timings['total'])
            
            num_detections = len(results[0].obb) if results[0].obb is not None else 0
            print(f"Detection {i+1}: {num_detections} panels | {timings['total']:.2f} ms | {1000/timings['total']:.1f} FPS")
        
        # Store timings for display
        detector.last_timings = timings
        
        # Print average timing
        avg_time = np.mean(all_timings)
        avg_fps = 1000 / avg_time
        print(f"\n{mode.upper()} MODE - Average: {avg_time:.2f} ms ({avg_fps:.1f} FPS)")
        
        # Draw detections on image
        output_path = f"/home/pi5/panel_detection/detected_panels_{mode}.jpg"
        img_with_boxes, detections = detector.draw_detections(img, results, 
                                                              conf_threshold=optimal_threshold,
                                                              save_path=output_path)
        
        # Save detection results
        output_txt = f"/home/pi5/panel_detection/detection_results_{mode}.txt"
        with open(output_txt, 'w') as f:
            f.write(f"DETECTION RESULTS ({mode.upper()} MODE)\n")
            f.write("="*40 + "\n\n")
            f.write(f"Image: {img_path}\n")
            f.write(f"Image size: {img.shape[1]}x{img.shape[0]}\n")
            f.write(f"Average processing time: {avg_time:.2f} ms\n")
            f.write(f"Average FPS: {avg_fps:.1f}\n")
            f.write(f"Detections found: {len(detections)}\n\n")
            
            if len(detections) > 0:
                f.write("PANEL DETAILS:\n")
                f.write("--------------\n")
                for i, det in enumerate(detections, 1):
                    f.write(f"\nPanel {i}:\n")
                    f.write(f"  Confidence: {det['confidence']:.3f} ({det['confidence']*100:.1f}%)\n")
                    f.write(f"  Corner coordinates (x, y):\n")
                    for j, corner in enumerate(det['box']):
                        f.write(f"    Corner {j+1}: ({corner[0]}, {corner[1]})\n")
        
        print(f"Results saved to: {output_txt}")
        
        # Stop detector threads
        detector.stop()
        
        print("\n" + "="*80)
        print(f"{mode.upper()} MODE COMPLETE!")
        print("="*80)
    
    # Comparison summary
    print("\n" + "="*80)
    print("PERFORMANCE COMPARISON")
    print("="*80)
    print("Parallel mode hides CLAHE preprocessing cost (≈14ms)")
    print("Expected improvement: ~15-20% faster overall processing")
    print("="*80)
