"""
Diagnostic: isolate CLAHE, preprocess_batch, and ONNX inference timing.
Run this on the Pi BEFORE making any more changes to the main pipeline.
"""
import cv2
import numpy as np
import time
import onnxruntime as ort
import glob

CLASSIFIER_ONNX_PATH = "/home/pi5/binary_class/ncnn_export/classifier_fp16.onnx"
TEST_CROPS_DIR       = "/home/pi5/binary_class/test_crops"   # from extract_test_crops.py
TEST_IMAGES_DIR      = "/home/pi5/binary_class/test_images"

_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(1, 3, 1, 1)
_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(1, 3, 1, 1)

def timer(fn, *args, runs=10):
    # Warmup
    for _ in range(3):
        fn(*args)
    times = []
    for _ in range(runs):
        t0 = time.perf_counter()
        fn(*args)
        times.append((time.perf_counter() - t0) * 1000)
    return min(times), sum(times)/len(times), max(times)

# ── 1. CLAHE timing ──────────────────────────────────────────────────────────
print("=" * 60)
print("1. CLAHE TIMING")
print("=" * 60)

img_paths = glob.glob(f"{TEST_IMAGES_DIR}/*.jpg")[:4]
for path in img_paths:
    img = cv2.imread(path)
    img_r = cv2.resize(img, (640, 640))
    
    # Method A: global CLAHE (what causes the spike)
    global_clahe = cv2.createCLAHE(clipLimit=1.5, tileGridSize=(4, 4))
    def clahe_global():
        lab = cv2.cvtColor(img_r, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        l = global_clahe.apply(l)
        cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_LAB2BGR)

    mn, avg, mx = timer(clahe_global, runs=20)
    print(f"  {path.split('/')[-1]:<30} min:{mn:.1f}ms  avg:{avg:.1f}ms  max:{mx:.1f}ms")

# ── 2. preprocess_batch timing ───────────────────────────────────────────────
print()
print("=" * 60)
print("2. PREPROCESS_BATCH TIMING (per batch size)")
print("=" * 60)

# Load real crops
crop_paths = sorted(glob.glob(f"{TEST_CROPS_DIR}/*.jpg"))
if not crop_paths:
    print("  No crops found — run extract_test_crops.py first")
    print("  Using synthetic crops instead")
    all_crops = [np.random.randint(0, 255, (80, 120, 3), dtype=np.uint8) for _ in range(8)]
else:
    all_crops = [cv2.imread(p) for p in crop_paths[:8]]
    all_crops = [c for c in all_crops if c is not None]

def preprocess_batch_v1(crops):
    """Current version — loop with transpose"""
    n = len(crops)
    batch = np.empty((n, 3, 224, 224), dtype=np.float32)
    for i, crop in enumerate(crops):
        resized = cv2.resize(crop, (224, 224))
        batch[i] = np.ascontiguousarray(resized[:, :, ::-1]).transpose(2, 0, 1)
    batch /= 255.0
    batch -= _MEAN
    batch /= _STD
    return batch

def preprocess_batch_v2(crops):
    """Alternative: avoid transpose by using copyMakeBorder + dstack"""
    n = len(crops)
    batch = np.empty((n, 3, 224, 224), dtype=np.float32)
    for i, crop in enumerate(crops):
        resized = cv2.resize(crop, (224, 224))
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB).astype(np.float32)
        rgb /= 255.0
        rgb -= np.array([0.485, 0.456, 0.406], dtype=np.float32)
        rgb /= np.array([0.229, 0.224, 0.225], dtype=np.float32)
        batch[i] = rgb.transpose(2, 0, 1)
    return batch

for n_panels in [2, 4, 6, 8]:
    crops = all_crops[:n_panels]
    if len(crops) < n_panels:
        crops = crops + [crops[-1]] * (n_panels - len(crops))
    
    mn1, avg1, _ = timer(preprocess_batch_v1, crops, runs=50)
    mn2, avg2, _ = timer(preprocess_batch_v2, crops, runs=50)
    print(f"  {n_panels} panels — v1(current): avg={avg1:.1f}ms  v2(alt): avg={avg2:.1f}ms")

# ── 3. Pure ONNX inference timing ────────────────────────────────────────────
print()
print("=" * 60)
print("3. PURE ONNX INFERENCE TIMING (no preprocessing)")
print("=" * 60)

sess_opts = ort.SessionOptions()
sess_opts.intra_op_num_threads = 2
sess_opts.inter_op_num_threads = 1
sess_opts.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
sess_opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
sess = ort.InferenceSession(CLASSIFIER_ONNX_PATH, sess_opts,
                             providers=["CPUExecutionProvider"])
inp_name = sess.get_inputs()[0].name

for n_panels in [1, 2, 4, 6, 8]:
    dummy = np.zeros((n_panels, 3, 224, 224), dtype=np.float32)
    def run_onnx():
        sess.run(None, {inp_name: dummy})
    mn, avg, mx = timer(run_onnx, runs=50)
    print(f"  batch={n_panels}  min:{mn:.1f}ms  avg:{avg:.1f}ms  max:{mx:.1f}ms")

# ── 4. Summary ───────────────────────────────────────────────────────────────
print()
print("=" * 60)
print("INTERPRETATION")
print("=" * 60)
print("If ONNX inference (section 3) is fast but classify in pipeline is slow,")
print("  → bottleneck is preprocess_batch (section 2)")
print("If preprocess_batch is fast but classify in pipeline is slow,")
print("  → bottleneck is thread contention or queue latency")
print("If CLAHE max >> CLAHE avg,")
print("  → global CLAHE object is still being used (thread safety issue)")
