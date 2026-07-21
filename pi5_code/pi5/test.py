import cv2

# Test RGB camera
print("Testing RGB camera (/dev/video2)...")
cap_rgb = cv2.VideoCapture("/dev/video2")
if cap_rgb.isOpened():
    ret, frame = cap_rgb.read()
    if ret:
        print(f"✓ RGB camera works! Frame shape: {frame.shape}")
    else:
        print("✗ Can open but cannot read frame")
    cap_rgb.release()
else:
    print("✗ Cannot open RGB camera")

# Test Thermal camera
print("\nTesting Thermal camera (/dev/video0)...")
cap_thermal = cv2.VideoCapture("/dev/video0")
if cap_thermal.isOpened():
    ret, frame = cap_thermal.read()
    if ret:
        print(f"✓ Thermal camera works! Frame shape: {frame.shape}")
    else:
        print("✗ Can open but cannot read frame")
    cap_thermal.release()
else:
    print("✗ Cannot open Thermal camera")
