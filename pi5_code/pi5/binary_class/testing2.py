# test_320.py — run on Pi
from ultralytics import YOLO
import cv2, glob

model416 = YOLO("/home/pi5/panel_detection2/best_ncnn_model_416", task="obb")
model640 = YOLO("/home/pi5/panel_detection2/best_ncnn_model", task="obb")

for path in glob.glob("/home/pi5/binary_class/test_images/*.jpg"):
    r416 = model416.predict(path, imgsz=416, conf=0.35, iou=0.45, verbose=False)
    r640 = model640.predict(path, imgsz=640, conf=0.35, iou=0.45, verbose=False)
    n416 = len(r416[0].obb.xyxyxyxy) if r416[0].obb else 0
    n640 = len(r640[0].obb.xyxyxyxy) if r640[0].obb else 0
    match = "✅" if n416 == n640 else "⚠️ MISMATCH"
    print(f"{match}  {path.split('/')[-1]}: 640→{n640} panels, 416→{n416} panels")
