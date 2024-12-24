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

    def detect_issues(self, img):
        """Detect various issues with the image inputted."""
        return {
            "Blurred": AnalyzeImage.is_this_image_wearing_glasses(img),
            "Contrast": AnalyzeImage.check_contrast(img),
            "SnP": AnalyzeImage.check_for_salt_and_pepper(img),
            "High Brightness": AnalyzeImage.check_if_its_sunbathing(img),
            "Low Brightness": AnalyzeImage.is_this_a_midnight_snack(img),
            "Frequency Discrepancies": AnalyzeImage.detect_periodic(img),
        }

    def apply_preprocessing(self, img, issues):
        """Apply corrections based on detected issues."""
        if issues["Frequency Discrepancies"]:
            img = ImageTransformer.periodic_noise_removal(img, 0.1)
        if issues["Blurred"]:
            img = ImageTransformer.apply_sharpening(img)
        if issues["High Brightness"]:
            img = PreprocessImage.gamma_correction(img, 30)
        if issues["Low Brightness"]:
            img = PreprocessImage.too_dark(img)
        if issues["SnP"]:
            img = PreprocessImage.remove_seasoning(img)
        if issues["Contrast"]:
            img = PreprocessImage.enhance_contrast(img)
        return img

    def handle_obstruction_flags(self, img, issues):
        """Handle flags for rotation and obstacles."""
        if AnalyzeImage.is_rotated(img, 90):
            img = PreprocessImage.rotate_by_contour(img)
            issues["Rotated"] = True

        if AnalyzeImage.spot_the_obstacle_course(img):
            img = PreprocessImage.clear_the_pathway(img)
            issues["Obstacle"] = True

        return img

    def finalize_processing(self, img):
        """Apply final image processing steps."""
        threshed = PreprocessImage.apply_threshold(img)
        cropped = PreprocessImage.remove_top_rows(
            PreprocessImage.crop_to_contour(threshed), 3
        )
        processed_img = cv2.morphologyEx(
            cropped, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (1, 2))
        )
        height = PreprocessImage.get_barcode_height(processed_img)
        processed_img = PreprocessImage.open_the_door(processed_img, height)
        processed_img = PreprocessImage.close_the_door(processed_img)
        threshed = PreprocessImage.apply_threshold(processed_img)
        return PreprocessImage.crop_to_contour(threshed)

    def save_image(self, img, image_path):
        """Save the processed image."""
        os.makedirs(self.OUTPUT_PATH, exist_ok=True)
        save_image_path = os.path.join(self.OUTPUT_PATH, os.path.basename(image_path))
        cv2.imwrite(save_image_path, img)
        return save_image_path

    def decode_barcode(self, img, image_name):
        """Decode barcode from image and save decoded digits."""
        barcode_decoder = BarcodeDecoder()
        decoded_digits = barcode_decoder.crack_the_code(img)
        decoded_digits_path = os.path.join(self.OUTPUT_PATH, "decoded_digits.txt")
        with open(decoded_digits_path, "a") as file:
            file.write(f"{image_name[:3]}: {' '.join(decoded_digits)}\n")
        return decoded_digits

    def save_results(self, image_name, decoded_digits, flags):
        """Save the flags in a file and decoded barcodes to another file."""
        result_file_path = os.path.join(self.OUTPUT_PATH, "flags_and_decoded.txt")
        os.makedirs(self.OUTPUT_PATH, exist_ok=True)
        flags_text = ", ".join(flags) if flags else "No flags"
        with open(result_file_path, "a") as result_file:
            result_file.write(
                f"{image_name}: Flags: {flags_text}, Decoded: {', '.join(decoded_digits)}\n"
            )
        if flags:
            print(f"{image_name}: {', '.join(flags)}")
        else:
            print(f"{image_name}: No flags")

    def process_image(self, image_path):
        """Process a single image.(made for the GUI)"""
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            print(f"Error reading image: {image_path}")
            return

        # Detect issues and preprocess the image
        issues = self.detect_issues(img)
        img = self.apply_preprocessing(img, issues)

        # Handle obstruction like rotation and object covering a small part of the barcode
        img = self.handle_obstruction_flags(img, issues)
        final_img = self.finalize_processing(img)

        # Save processed image for later use(maybe there would be)
        image_name = os.path.basename(image_path)
        self.save_image(final_img, image_path)

        # Decode & Save the Decoded Barcode digits to decoded_digits.txt
        decoded_digits = self.decode_barcode(final_img, image_name)
        triggered_flags = [key for key, value in issues.items() if value]
        self.save_results(image_name, decoded_digits, triggered_flags)

    def process_all_images(self):
        """Process all images in the provided list.(Made for Testing
        the Batch of Test Cases that is given to test our code"""
        for image_path in self.image_paths:
            self.process_image(image_path)