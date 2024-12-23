import os
import cv2
from ImageTransformerUtils import ImageTransformer
from PreprocessImageUtils import PreprocessImage
from BarcodeDecoder import BarcodeDecoder
from AnalyzeImageUtils import AnalyzeImage


class ImageProcessor:
    OUTPUT_PATH = 'processed'

    def __init__(self, image_paths):
        self.image_paths = image_paths

    def process_image(self, image_path):
        # Read the image in grayscale mode
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            print(f"Error reading image: {image_path}")
            return

        # Detect various image issues
        blur_flag = AnalyzeImage.is_this_image_wearing_glasses(img)
        contrast_flag = AnalyzeImage.check_contrast(img)
        salt_pepper_flag = AnalyzeImage.check_for_salt_and_pepper(img)
        high_brightness_flag = AnalyzeImage.check_if_its_sunbathing(img)
        low_brightness_flag = AnalyzeImage.is_this_a_midnight_snack(img)
        periodic_flag = AnalyzeImage.detect_periodic(img)

        # Preprocess image based on detected issues
        processed_img = img.copy()

        if periodic_flag:
            processed_img = ImageTransformer.periodic_noise_removal(processed_img, 0.1)
        if blur_flag:
            processed_img = ImageTransformer.apply_sharpening(processed_img)
        if high_brightness_flag:
            processed_img = PreprocessImage.gamma_correction(processed_img, 30)
        if low_brightness_flag:
            processed_img = PreprocessImage.too_dark(processed_img)
        if salt_pepper_flag:
            processed_img = PreprocessImage.remove_seasoning(processed_img)
        if contrast_flag:
            processed_img = PreprocessImage.enhance_contrast(processed_img)

        # Handle rotation correction
        is_rotated_flag = AnalyzeImage.is_rotated(processed_img, 90)
        if is_rotated_flag:
            processed_img = PreprocessImage.rotate_by_contour(processed_img)

        # Handle obstacle removal
        obstacle_flag = AnalyzeImage.spot_the_obstacle_course(processed_img)
        if obstacle_flag:
            processed_img = PreprocessImage.clear_the_pathway(processed_img)

        # Final processing steps
        threshed = PreprocessImage.apply_threshold(processed_img)
        contoured = PreprocessImage.crop_to_contour(threshed)
        cropped = PreprocessImage.remove_top_rows(contoured, 3)

        # Additional processing steps
        processed_img = cv2.morphologyEx(cropped, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (1, 2)))
        height = PreprocessImage.get_barcode_height(processed_img)
        processed_img = PreprocessImage.open_the_door(processed_img, height)
        processed_img = PreprocessImage.close_the_door(processed_img)
        threshed = PreprocessImage.apply_threshold(processed_img)
        final_img = PreprocessImage.crop_to_contour(threshed)

        # Save the processed image
        save_path = self.OUTPUT_PATH
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        save_image_path = os.path.join(save_path, os.path.basename(image_path))
        cv2.imwrite(save_image_path, final_img)

        # Decode the barcode
        barcode_decoder = BarcodeDecoder()
        decoded_digits = barcode_decoder.crack_the_code(final_img)

        # Append the decoded digits to a text file
        with open(rf"{self.OUTPUT_PATH}/decoded_digits.txt", "a") as file:
            file.write(f"{os.path.basename(image_path)[:3]}: {' '.join(decoded_digits)}\n")

        # Collect flags
        flags = []
        if blur_flag:
            flags.append('Blur')
        if contrast_flag:
            flags.append('Contrast')
        if salt_pepper_flag:
            flags.append('Salt & Pepper')
        if high_brightness_flag:
            flags.append('High Brightness')
        if low_brightness_flag:
            flags.append('Low Brightness')
        if periodic_flag:
            flags.append('Periodic')
        if is_rotated_flag:
            flags.append('Rotated')
        if obstacle_flag:
            flags.append('Obstacle')

        # Print flags next to the image name
        if flags:
            print(f"{os.path.basename(image_path)}: {', '.join(flags)}")
        else:
            print(f"{os.path.basename(image_path)}: No flags")

    def process_all_images(self):
        for image_path in self.image_paths:
            self.process_image(image_path)
