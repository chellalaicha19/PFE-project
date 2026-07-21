# test_on_images_gradcam.py
import tensorflow as tf
import numpy as np
import cv2
from pathlib import Path

# ── Load model ────────────────────────────────────────────────────────────────
model = tf.keras.models.load_model('/Users/mac/Documents/PFE/thermal_binary_class/thermal_binary_mobilenetv2.keras')

# ── Build Grad-CAM model ──────────────────────────────────────────────────────
# Conv_1 lives inside the mobilenetv2 sub-model (layer index 1).
# We must route through the sub-model so the graph stays connected:
#   outer_input → mobilenetv2_submodel → [Conv_1 output, submodel output]
#                                                              ↓
#                                          GAP → BN → Dropout → Dense → output

base_model = model.get_layer('mobilenetv2_1.00_224')   # the nested Functional

# Sub-model: outer input  →  (conv_1_out, base_out)
conv1_layer  = base_model.get_layer('Conv_1')
base_grad_model = tf.keras.models.Model(
    inputs  = base_model.input,
    outputs = [conv1_layer.output, base_model.output]
)

# We'll call base_grad_model inside GradientTape manually, so we only need
# the outer head layers that come after the base model.
# Collect them in order: GAP → BN → Dropout → Dense → Dropout → Dense
head_layers = model.layers[2:]   # everything after the Functional sub-model
print(f"Grad-CAM target : mobilenetv2_1.00_224 → Conv_1")
print(f"Head layers      : {[l.name for l in head_layers]}\n")


def compute_gradcam(img_array):
    """
    Returns a normalised [0,1] float32 heatmap of shape (H, W).
    img_array: preprocessed (1, 224, 224, 3) float32 tensor
    """
    img_tensor = tf.cast(img_array, tf.float32)

    with tf.GradientTape() as tape:
        # Forward through base model — tape watches the conv output
        conv_outputs, base_out = base_grad_model(img_tensor, training=False)
        tape.watch(conv_outputs)

        # Forward through the head
        x = base_out
        for layer in head_layers:
            x = layer(x, training=False)
        predictions = x                    # shape (1, 1)

        loss = predictions[:, 0]           # scalar

    grads = tape.gradient(loss, conv_outputs)   # (1, h, w, c)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))  # (c,)

    heatmap = conv_outputs[0] @ pooled_grads[..., tf.newaxis]  # (h, w, 1)
    heatmap = tf.squeeze(heatmap).numpy()

    # ReLU + normalise
    heatmap = np.maximum(heatmap, 0)
    if heatmap.max() > 0:
        heatmap /= heatmap.max()
    return heatmap.astype(np.float32)


def overlay_gradcam(original_bgr, heatmap, alpha=0.45):
    """Resize heatmap to image size and blend with the original."""
    h, w = original_bgr.shape[:2]
    heatmap_resized = cv2.resize(heatmap, (w, h))
    heatmap_uint8   = np.uint8(255 * heatmap_resized)
    colored         = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
    return cv2.addWeighted(original_bgr, 1 - alpha, colored, alpha, 0)


def predict_with_gradcam(image_path, output_dir=Path("gradcam_results")):
    output_dir.mkdir(exist_ok=True)
    image_path = Path(image_path)

    # Load original (full resolution, BGR)
    original_bgr = cv2.imread(str(image_path))
    if original_bgr is None:
        print(f"  [skip] Cannot read: {image_path}")
        return None

    # Preprocess for model
    img       = tf.keras.utils.load_img(image_path, target_size=(224, 224))
    img_array = tf.keras.utils.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0)
    img_array = tf.keras.applications.mobilenet_v2.preprocess_input(img_array)

    # Prediction
    prediction = float(model.predict(img_array, verbose=0)[0][0])
    if prediction < 0.5:
        label, confidence = "ANOMALY",    1 - prediction
    else:
        label, confidence = "NO_ANOMALY", prediction

    # Grad-CAM
    heatmap  = compute_gradcam(img_array)
    overlaid = overlay_gradcam(original_bgr, heatmap)

    # Annotate overlay panel
    color = (0, 0, 255) if label == "ANOMALY" else (0, 200, 0)
    text  = f"{label}  {confidence:.1%}"
    cv2.putText(overlaid, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,0,0), 3)
    cv2.putText(overlaid, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color,   2)

    # Side-by-side: original | grad-cam overlay
    h, w = original_bgr.shape[:2]
    combined = np.hstack([original_bgr, cv2.resize(overlaid, (w, h))])

    out_path = output_dir / f"{image_path.stem}_gradcam.jpg"
    cv2.imwrite(str(out_path), combined)

    print(f"  {label:<12}  conf={confidence:.3f}  raw={prediction:.3f}  → {out_path.name}")
    return label, confidence, prediction


# ── Run ───────────────────────────────────────────────────────────────────────
test_images = [
    '/Users/mac/Documents/PFE/test_rgb+thermal/test1_thermal.png',
    '/Users/mac/Documents/PFE/test_rgb+thermal/test2_thermal.png',
    '/Users/mac/Documents/PFE/test_rgb+thermal/test3_thermal.png',
    '/Users/mac/Documents/PFE/test_rgb+thermal/test4_thermal.png',
    '/Users/mac/Documents/PFE/test_rgb+thermal/test5_thermal.png'
    
]

print("Running Grad-CAM inference...\n")
for p in test_images:
    if Path(p).exists():
        predict_with_gradcam(p)
    else:
        print(f"  [missing] {p}")

print("\nDone. Results saved to ./gradcam_results/")