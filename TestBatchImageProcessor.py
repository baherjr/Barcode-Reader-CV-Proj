import os
from ImageProcessor import ImageProcessor

# Specify the paths for test images and output directory
TEST_PATH = 'tests'
OUTPUT_PATH = 'processed'


def main():
    # Check if the test path exists
    if not os.path.exists(TEST_PATH):
        print(f"Error: Test path '{TEST_PATH}' does not exist.")
        return

    # Collect all image file paths in the test directory
    image_paths = [os.path.join(TEST_PATH, filename) for filename in os.listdir(TEST_PATH)
                   if filename.lower().endswith(('.jpg', '.jpeg', '.png'))]

    # Ensure there are images to process
    if not image_paths:
        print(f"No image files found in '{TEST_PATH}'.")
        return

    # Pass the collected image paths to the ImageProcessor
    processor = ImageProcessor(image_paths)

    print("Starting image processing...")
    # Process each image
    for image_path in image_paths:
        processor.process_image(image_path)
    print("Image processing completed.")

    # Provide output path information
    print(f"Processed images and decoded barcodes are saved in '{OUTPUT_PATH}'.")


# This allows the script to be run directly
if __name__ == "__main__":
    main()
