import os
import cv2
from ImageTransformerUtils import ImageTransformer
from PreprocessImageUtils import PreprocessImage
from BarcodeDecoder import BarcodeDecoder
from AnalyzeImageUtils import AnalyzeImage


class ImageProcessor:
    OUTPUT_PATH = 'processed_barcodes'

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
        contrast_flag = AnalyzeImage.detect_contrast(img)
        salt_pepper_flag = AnalyzeImage.check_for_salt_and_pepper(img)
        high_brightness_flag = AnalyzeImage.check_if_its_sunbathing(img)
        low_brightness_flag = AnalyzeImage.is_this_a_midnight_snack(img)
        periodic_flag = AnalyzeImage.detect_periodic(img)

        print(f"Image: {image_path}")
        print(f"Blur: {blur_flag}")
        print(f"Contrast: {contrast_flag}")
        print(f"Salt & Pepper: {salt_pepper_flag}")
        print(f"High Brightness: {high_brightness_flag}")
        print(f"Low Brightness: {low_brightness_flag}")
        print(f"Periodic: {periodic_flag}")

        # Preprocess image based on detected issues
        processed_img = img.copy()

        if periodic_flag:
            processed_img = ImageTransformer.periodic_noise_removal(processed_img, 0.1)
        if blur_flag:
            processed_img = ImageTransformer.Sharpen(processed_img)
        if high_brightness_flag:
            processed_img = PreprocessImage.gamma_correction(processed_img, 30)
        if low_brightness_flag:
            processed_img = PreprocessImage.process_dark_barcode(processed_img)
        if salt_pepper_flag:
            processed_img = PreprocessImage.sheel_mal7_wfelfel(processed_img)
        if contrast_flag:
            processed_img = PreprocessImage.la2ena_el_contrast(processed_img)

        # Handle rotation correction
        is_rotated_flag = AnalyzeImage.is_rotated(processed_img, 90)
        print(f"Rotated: {is_rotated_flag}")
        if is_rotated_flag:
            processed_img = PreprocessImage.contour_rotated(processed_img)

        # Handle obstacle removal
        obstacle_flag = AnalyzeImage.obstacle_detection(processed_img)
        print(f"Obstacle: {obstacle_flag}")
        if obstacle_flag:
            processed_img = PreprocessImage.remove_Obstacle(processed_img)

        # Final processing steps
        threshed = PreprocessImage.thresholding(processed_img)
        contoured = PreprocessImage.contour(threshed)
        cropped = PreprocessImage.crop_rows(contoured, 3)

        # Additional processing steps
        processed_img = cv2.morphologyEx(cropped, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (1, 2)))
        height = PreprocessImage.calculate_vertical_bar_height(processed_img)
        processed_img = PreprocessImage.apply_opening(processed_img, height)
        processed_img = PreprocessImage.apply_closing(processed_img)
        threshed = PreprocessImage.thresholding(processed_img)
        final_img = PreprocessImage.contour(threshed)

        # Save the processed image
        save_path = self.OUTPUT_PATH
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        save_image_path = os.path.join(save_path, os.path.basename(image_path))
        cv2.imwrite(save_image_path, final_img)

        # Decode the barcode
        barcode_decoder = BarcodeDecoder()
        decoded_digits = barcode_decoder.decode_barcode(final_img)
        print("Decoded digits:", decoded_digits)

        # Append the decoded digits to a text file
        with open(rf"{self.OUTPUT_PATH}/decoded_digits.txt", "a") as file:
            file.write(f"{os.path.basename(image_path)[:3]}: {' '.join(decoded_digits)}\n")

