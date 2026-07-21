import cv2
import numpy as np
import time

paths = [
    "/home/pi5/binary_class/test_images/test_image13.jpg",
    "/home/pi5/binary_class/test_images/test_image14.jpg",
    "/home/pi5/binary_class/test_images/test_image4.jpg",
]

clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))

for p in paths:
    img = cv2.imread(p)
    print(f"Loaded: {img.shape}")
    
    # Time resize separately
    t0 = time.perf_counter()
    img_r = cv2.resize(img, (640, 640))
    t_resize = (time.perf_counter() - t0) * 1000

    # Time CLAHE separately
    t0 = time.perf_counter()
    lab = cv2.cvtColor(img_r, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    l = clahe.apply(l)
    result = cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_LAB2BGR)
    t_clahe = (time.perf_counter() - t0) * 1000

    print(f"  Resize: {t_resize:.1f}ms  CLAHE: {t_clahe:.1f}ms  Total: {t_resize+t_clahe:.1f}ms")
