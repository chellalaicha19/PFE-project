from PIL import Image

def combine_images_side_by_side(image_path1, image_path2, output_path):
    # Open the two images
    img1 = Image.open("/Users/mac/Documents/good/fly/frame_000089/rgb.jpg")
    img2 = Image.open("/Users/mac/Documents/good/fly/frame_000089/thermal.jpg")

    # Get the dimensions of both images
    w1, h1 = img1.size
    w2, h2 = img2.size

    # Calculate the size of the new image
    new_width = w1 + w2
    new_height = max(h1, h2)

    # Create a new blank image with the appropriate size
    combined_img = Image.new('RGB', (new_width, new_height))

    # Paste the original images into the new blank image
    combined_img.paste(img1, (0, 0))
    combined_img.paste(img2, (w1, 0))

    # Save the final combined image
    combined_img.save(output_path)
    print(f"Images combined successfully! Saved as: {output_path}")

# Run the function with your specific file names
combine_images_side_by_side("annotated_3.jpg", "thermal_3.jpg", "combined_result.jpg")