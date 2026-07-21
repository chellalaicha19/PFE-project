from PIL import Image

# 1. Open the two images
img1 = Image.open("/Users/mac/Documents/PFE/fusion/testing/test4_rgb.png")
img2 = Image.open("/Users/mac/Documents/PFE/fusion/testing/test4_thermal.png")

# 2. Create a canvas large enough to hold both side-by-side
total_width = img1.width + img2.width
max_height = max(img1.height, img2.height)
combined_image = Image.new("RGB", (total_width, max_height))

# 3. Paste the images onto the canvas
combined_image.paste(img1, (0, 0))  # Left image
combined_image.paste(img2, (img1.width, 0))  # Right image

# 4. Save the result
combined_image.save("combined_output.jpg")