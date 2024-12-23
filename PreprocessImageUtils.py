import numpy as np
import cv2
import math
from scipy.signal import find_peaks



class PreprocessImage:

    @staticmethod
    def thresholding(image, value=127):
        ret, thresh = cv2.threshold(image, value, 255, cv2.THRESH_BINARY)
        return thresh

    @staticmethod
    def contour(image):
        inv = cv2.bitwise_not(image)
        x, y, w, h = cv2.boundingRect(inv)

        if y == 0 and x == 0 and h == image.shape[0] and w == image.shape[1]:
            return image
        crop = image[y:y + h - h // 4, x:x + w]
        return crop

    def calculate_vertical_bar_height(image):
        # Use the static method `PreprocessImage.contour` directly
        contoured = PreprocessImage.contour(image)
        if len(contoured.shape) == 2:
            height, _ = contoured.shape
        else:
            height, _, _ = contoured.shape
        return height

    @staticmethod
    def morphology_operation(image, operation, kernel_size):
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, kernel_size)
        return cv2.morphologyEx(image, operation, kernel)

    @staticmethod
    def apply_closing(image):
        return PreprocessImage.morphology_operation(image, cv2.MORPH_CLOSE, (3, 3))

    @staticmethod
    def apply_opening(image, bar_height):
        return PreprocessImage.morphology_operation(image, cv2.MORPH_OPEN, (1,bar_height))

    @staticmethod
    def apply_gaussian(image, count=1):
        gaussian_kernel = np.array([[1, 2, 1],
                                    [2, 4, 2],
                                    [1, 2, 1]], dtype=np.float32)
        gaussian_kernel /= np.sum(gaussian_kernel)

        filtered_image = image
        for _ in range(count):
            filtered_image = cv2.filter2D(filtered_image, -1, gaussian_kernel)

        return filtered_image

    @staticmethod
    def contour_rotated(img):
        padded_image = cv2.copyMakeBorder(img, 100, 100, 200, 200, cv2.BORDER_CONSTANT, value=[255, 255, 255])
        ret, thresh = cv2.threshold(padded_image, 127, 255, 0)

        contours, _ = cv2.findContours(thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

        selected_contour = None
        for contour in contours:
            area = cv2.contourArea(contour)
            if 500 <= area <= 800:
                selected_contour = contour
                break

        if selected_contour is None:
            raise ValueError(f"No contour found within the specified area range: {400}-{600}.")

        rect = cv2.minAreaRect(selected_contour)
        angle = rect[2]

        (height, width) = thresh.shape[:2]
        center = (width // 2, height // 2)

        if angle < -45:
            angle += 90
        rotation_matrix = cv2.getRotationMatrix2D(center, angle - 90, 1.0)
        corrected_image = cv2.warpAffine(thresh, rotation_matrix, (width, height), flags=cv2.INTER_CUBIC,
                                         borderMode=cv2.BORDER_CONSTANT, borderValue=(255, 255, 255))

        return corrected_image

    @staticmethod
    def remove_Obstacle(gray):
        gray[(gray >= 20) & (gray <= 220)] = 255
        gray = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2)))
        threshed = PreprocessImage.thresholding(gray)
        contoured = PreprocessImage.contour(threshed)
        processed_img = cv2.morphologyEx(contoured, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (1, 200)))
        return processed_img

    @staticmethod
    def sheel_mal7_wfelfel(image):
        blurred_image = cv2.blur(image, (1, 15))
        filtered_image = cv2.medianBlur(blurred_image, 5)
        _, filtered_thresh = cv2.threshold(filtered_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        filtered_thresh = cv2.morphologyEx(filtered_thresh, cv2.MORPH_CLOSE,
                                           cv2.getStructuringElement(cv2.MORPH_RECT, (4, 4)))
        return filtered_thresh

    @staticmethod
    def gamma_correction(image, gamma):
        lookup_table = np.empty((1, 256), np.uint8)
        for i in range(256):
            lookup_table[0, i] = np.clip(pow(i / 255.0, gamma) * 255.0, 0, 255)

        corrected_image = cv2.LUT(image, lookup_table)
        threshed = PreprocessImage.thresholding(corrected_image)
        return threshed

    @staticmethod
    def process_dark_barcode(image):
        _, binary_image = cv2.threshold(image, 200, 255, cv2.THRESH_BINARY_INV)

        inpainted_image = cv2.inpaint(image, binary_image, inpaintRadius=3, flags=cv2.INPAINT_TELEA)

        _, bright_mask = cv2.threshold(inpainted_image, 10, 255, cv2.THRESH_BINARY)

        lightened_image = np.where(bright_mask == 255,
                                   np.clip(inpainted_image * 4, 0, 255).astype(np.uint8),
                                   inpainted_image)

        _, binary_result = cv2.threshold(lightened_image, 50, 255, cv2.THRESH_BINARY)

        bar_height = PreprocessImage.calculate_vertical_bar_height(binary_result)

        closed_image = PreprocessImage.apply_closing(binary_result)
        opened_image = PreprocessImage.apply_opening(closed_image, bar_height)
        threshed = PreprocessImage.thresholding(opened_image)

        return threshed

    @staticmethod
    def la2ena_el_contrast(image):
        histogram = cv2.calcHist([image], [0], None, [256], [0, 256])
        histogram = histogram / histogram.sum()

        histogram_diff = np.diff(histogram.ravel())
        peaks, _ = find_peaks(histogram_diff)
        peak_values = histogram_diff[peaks]

        highest_peaks_indices = peaks[np.argsort(peak_values)[-2:]]
        midpoint_intensity = np.mean(highest_peaks_indices)

        _, threshed = cv2.threshold(image, math.ceil(midpoint_intensity), 255, cv2.THRESH_BINARY)
        return threshed

    @staticmethod
    def crop_rows(image, num_rows=2):
        """
        Crop the top N rows of the image.
        """
        if image.shape[0] <= num_rows:
            raise ValueError("Image has fewer rows than the number of rows to crop.")
        cropped_image = image[num_rows:, :]
        return cropped_image

    @staticmethod
    def crop_col(image, left_crop=0, right_crop=0):
        """
        Crop the specified number of columns from the left and right sides of the image.
        """
        rows, cols = image.shape[:2]

        left_crop = max(0, left_crop)
        right_crop = max(0, right_crop)
        if left_crop + right_crop >= cols:
            raise ValueError("The total crop (left + right) exceeds the image width.")

        cropped_image = image[:, left_crop:cols - right_crop]
        return cropped_image




