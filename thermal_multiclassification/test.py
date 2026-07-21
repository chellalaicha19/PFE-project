import numpy as np
import tensorflow as tf
from PIL import Image

# Path to your TFLite model
MODEL_PATH = "/Users/mac/Documents/PFE/thermal_multiclassification/thermal_multiclass_efficientnetb1_f16.tflite"

# Your test images
test_images = [
    '/Users/mac/Documents/PFE/test_rgb+thermal/test1_thermal.png',
    '/Users/mac/Documents/PFE/test_rgb+thermal/test2_thermal.png',
    '/Users/mac/Documents/PFE/test_rgb+thermal/test3_thermal.png',
    '/Users/mac/Documents/PFE/test_rgb+thermal/test4_thermal.png',
    '/Users/mac/Documents/PFE/test_rgb+thermal/test5_thermal.png',
    '/Users/mac/Documents/PFE/test_rgb+thermal/test6_thermal.png',
    '/Users/mac/Documents/PFE/test_rgb+thermal/test7_thermal.png',
    '/Users/mac/Documents/PFE/test_rgb+thermal/test8_thermal.png'
]

class_names = ['hotspot', 'partial_cold', 'full_cold']

def efficientnet_preprocess(image_array):
    """
    Replicates tf.keras.applications.efficientnet.preprocess_input
    Converts RGB to BGR and normalizes using ImageNet means
    """
    # Convert RGB to BGR (EfficientNet expects BGR)
    image_array = image_array[..., ::-1]
    
    # Subtract ImageNet means
    mean = [103.939, 116.779, 123.68]  # BGR order
    image_array[..., 0] -= mean[0]
    image_array[..., 1] -= mean[1]
    image_array[..., 2] -= mean[2]
    
    return image_array

def load_and_preprocess_image(image_path, target_size=(240, 240)):
    """Load and preprocess image for EfficientNetB1"""
    # Load image as RGB
    img = Image.open(image_path).convert('RGB')
    img = img.resize(target_size, Image.Resampling.BILINEAR)
    
    # Convert to numpy array (0-255 range)
    img_array = np.array(img, dtype=np.float32)
    
    # Apply EfficientNet preprocessing (RGB to BGR + mean subtraction)
    img_array = efficientnet_preprocess(img_array)
    
    # Add batch dimension
    img_array = np.expand_dims(img_array, axis=0)
    
    return img_array

# Load TFLite model
interpreter = tf.lite.Interpreter(model_path=MODEL_PATH)
interpreter.allocate_tensors()

# Get input and output details
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

print("Model loaded successfully!")
print(f"Input shape: {input_details[0]['shape']}")
print(f"Input dtype: {input_details[0]['dtype']}")
print()

# Test each image
print("Testing with CORRECT EfficientNet preprocessing:\n")
for img_path in test_images:
    try:
        input_data = load_and_preprocess_image(img_path)
        
        # Make sure dtype matches
        if input_details[0]['dtype'] == np.uint8:
            input_data = input_data.astype(np.uint8)
        else:
            input_data = input_data.astype(np.float32)
        
        # Set input tensor
        interpreter.set_tensor(input_details[0]['index'], input_data)
        
        # Run inference
        interpreter.invoke()
        
        # Get prediction
        predictions = interpreter.get_tensor(output_details[0]['index'])[0]
        
        # Get top prediction
        predicted_class_idx = np.argmax(predictions)
        confidence = predictions[predicted_class_idx] * 100
        
        print(f"Image: {img_path.split('/')[-1]}")
        print(f"Predictions: hotspot={predictions[0]:.3f}, partial_cold={predictions[1]:.3f}, full_cold={predictions[2]:.3f}")
        print(f"Predicted: {class_names[predicted_class_idx]} ({confidence:.2f}%)")
        print("-" * 50)
        
    except Exception as e:
        print(f"Error processing {img_path}: {e}")