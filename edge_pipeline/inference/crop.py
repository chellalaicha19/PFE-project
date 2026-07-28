# extract_test_crops.py — run on Pi once
from ultralytics import YOLO
import cv2, os, glob

model = YOLO("/home/pi5/panel_detection2/best_ncnn_model", task="obb")
os.makedirs("test_crops", exist_ok=True)

for path in glob.glob("test_images/*.jpg"):
    img = cv2.imread(path)
    img_r = cv2.resize(img, (640, 640))
    results = model.predict(img_r, imgsz=640, conf=0.35, iou=0.45, verbose=False)
    h, w = img_r.shape[:2]
    for r in results:
        if r.obb is None: continue
        for i, box in enumerate(r.obb.xyxyxyxy.cpu().numpy()):
            pts = box.reshape(4, 2)
            x1, y1 = int(pts[:,0].min()), int(pts[:,1].min())
            x2, y2 = int(pts[:,0].max()), int(pts[:,1].max())
            x1,y1,x2,y2 = max(0,x1),max(0,y1),min(w,x2),min(h,y2)
            crop = img_r[y1:y2, x1:x2]
            name = f"{os.path.basename(path)[:-4]}_panel{i+1}.jpg"
            cv2.imwrite(f"test_crops/{name}", crop)

print(f"Saved {len(glob.glob('test_crops/*.jpg'))} crops")
