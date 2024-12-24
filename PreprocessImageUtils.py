import numpy as np
import cv2
import math
from scipy.signal import find_peaks
from scipy import fftpack


class PreprocessImage:
    GLOBAL_THRESHOLD =  127

    @staticmethod
    def crop_to_contour(image):
        bounding_x, bounding_y, bounding_width, bounding_height = cv2.boundingRect(cv2.bitwise_not(image))

        if (bounding_x, bounding_y, bounding_width, bounding_height) == (0, 0, image.shape[1], image.shape[0]):
            return image

        return image[bounding_y: bounding_y + int(bounding_height * 0.75), bounding_x: bounding_x + bounding_width]

    @staticmethod
    def morphology_operation(image, operation, kernel_size):
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, kernel_size)
        return cv2.morphologyEx(image, operation, kernel)

    @staticmethod
    def close_the_door(image):
        """(3,3) kernel is the best because it works well with decode the code11 using width"""
        return PreprocessImage.morphology_operation(image, cv2.MORPH_CLOSE, (3, 3))

    @staticmethod
    def open_the_door(image, bar_height):
        """ (1,bar_height) kernel is used here to aligned with the height of the barcode region to process
        the bars vertically without affecting their width."""
        return PreprocessImage.morphology_operation(image, cv2.MORPH_OPEN, (1,bar_height))

    @staticmethod
    def gaussian_blur(image, iterations=1, kernel_size=(3, 1)):
        """
        Applies a Gaussian blur to the input image with a specified kernel size
        and number of iterations. Optimized for Code 11 barcodes.
        """
        if not isinstance(kernel_size, tuple) or len(kernel_size) != 2:
            raise ValueError("Kernel size must be a tuple of two integers, e.g., (3, 1).")

        # Create a Gaussian kernel programmatically
        gaussian_kernel = cv2.getGaussianKernel(kernel_size[0], sigma=-1)  # Create a 1D vertical Gaussian kernel
        gaussian_kernel = np.outer(gaussian_kernel, np.ones(kernel_size[1]))  # Extend to 2D based on kernel_size

        # Normalize the kernel
        gaussian_kernel /= np.sum(gaussian_kernel)

        # Apply the Gaussian blur for the specified number of iterations
        blurred_image = image
        for _ in range(iterations):
            blurred_image = cv2.filter2D(blurred_image, -1, gaussian_kernel)

        return blurred_image

    @staticmethod
    def rotate_by_contour(img , thresh_value=GLOBAL_THRESHOLD):
        padded_image = cv2.copyMakeBorder(img, 100, 100, 200, 200, cv2.BORDER_CONSTANT, value=[255, 255, 255])
        ret, thresh = cv2.threshold(padded_image, thresh=thresh_value, maxval=255, type=0)

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
    def apply_threshold(image, value=GLOBAL_THRESHOLD):
        ret, thresh = cv2.threshold(image, value, 255, cv2.THRESH_BINARY)
        return thresh

    @staticmethod
    def pad_image_for_rotation(image):
        """
        Pads the input image
        """
        return cv2.copyMakeBorder(image, 100, 100, 200, 200, cv2.BORDER_CONSTANT, value=[255, 255, 255])

    @staticmethod
    def get_contour_within_area(thresh_image, min_area, max_area):

        contours, _ = cv2.findContours(thresh_image, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            area = cv2.contourArea(contour)
            if min_area <= area <= max_area:
                return contour
        raise ValueError(f"No contour found within the specified area range: {min_area}-{max_area}.")

    @staticmethod
    def get_rotation_angle(contour):
        rect = cv2.minAreaRect(contour)
        angle = rect[2]
        return angle + 90 if angle < -45 else angle

    @staticmethod
    def apply_rotation(image, angle):
        (height, width) = image.shape[:2]
        center = (width // 2, height // 2)
        rotation_matrix = cv2.getRotationMatrix2D(center, angle - 90, 1.0)
        rotated_image = cv2.warpAffine(
            image, rotation_matrix, (width, height),
            flags=cv2.INTER_CUBIC,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(255, 255, 255)
        )
        return rotated_image

    @staticmethod
    def clear_the_pathway(gray):
        """
        Processes the input grayscale image to clear artifacts and prepare for further preprocessing.
        """
        gray[np.logical_and(gray >= 30, gray <= 220)] = 255
        threshed = PreprocessImage.apply_threshold(PreprocessImage.close_the_door(gray))
        contoured = PreprocessImage.crop_to_contour(threshed)
        bar_height = PreprocessImage.get_barcode_height(contoured)  # Kernel size set based on domain requirements (barcode height)
        processed_img = PreprocessImage.open_the_door(contoured, bar_height)
        return processed_img

    @staticmethod
    def remove_seasoning(image):
        # Apply vertical blur to the image
        vertically_blurred = cv2.blur(image, ksize=(1, 15))

        # Apply median filtering to reduce noise
        median_filtered = cv2.medianBlur(vertically_blurred, ksize=5)

        # Apply Otsu thresholding for binary segmentation
        _, otsu_thresholded = cv2.threshold(
            median_filtered, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )

        # Perform morphological closing to fill small gaps
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, ksize=(4, 4))
        morphologically_closed = cv2.morphologyEx(otsu_thresholded, cv2.MORPH_CLOSE, kernel)

        return morphologically_closed

    @staticmethod
    def gamma_correction(image, gamma):
        lookup_table = (np.arange(256) / 255.0) ** gamma * 255.0
        lookup_table = np.clip(lookup_table, 0, 255).astype(np.uint8)

        corrected_image = cv2.LUT(image, lookup_table)
        # Apply thresholding
        threshed = PreprocessImage.apply_threshold(corrected_image)

        return threshed

    def get_barcode_height(image):
        # Uses crop_to_contour to get the barcode heights
        contoured = PreprocessImage.crop_to_contour(image)
        if len(contoured.shape) == 2:
            height, _ = contoured.shape
        else:
            height, _, _ = contoured.shape
        return height

    @staticmethod
    def too_dark(image):
        _, binary_image = cv2.threshold(image, 200, 255, cv2.THRESH_BINARY_INV)
        inpainted_image = cv2.inpaint(image, binary_image, inpaintRadius=3, flags=cv2.INPAINT_TELEA)

        bright_mask = cv2.threshold(inpainted_image, 10, 255, cv2.THRESH_BINARY)[1]
        lightened_image = np.where(
            bright_mask == 255,
            np.clip(inpainted_image * 4, 0, 255).astype(np.uint8),
            inpainted_image
        )

        binary_result = cv2.threshold(lightened_image, 50, 255, cv2.THRESH_BINARY)[1]
        bar_height = PreprocessImage.get_barcode_height(binary_result)

        return PreprocessImage.apply_threshold(
            PreprocessImage.open_the_door(
                PreprocessImage.close_the_door(binary_result), bar_height
            )
        )

    @staticmethod
    def enhance_contrast(image):
        histogram = cv2.calcHist([image], [0], None, [256], [0, 256]).ravel()
        histogram /= histogram.sum()
        histogram_diff = np.diff(histogram)
        # Find peaks in the histogram difference
        peaks, _ = find_peaks(histogram_diff)
        if len(peaks) == 0:  # Handle rare case where no peaks are found
            return image
        top_peaks = peaks[np.argsort(histogram_diff[peaks])[-2:]]
        midpoint_intensity = np.mean(top_peaks)
        _, thresholded_image = cv2.threshold(image, math.ceil(midpoint_intensity), 255, cv2.THRESH_BINARY)
        return thresholded_image

    @staticmethod
    def remove_top_rows(image, num_rows=2):
        if image.shape[0] <= num_rows:
            raise ValueError("remove_top_rows error")
        cropped_image = image[num_rows:, :]
        return cropped_image

    @staticmethod
    def trim_sides(image, left_crop=0, right_crop=0):
        left_crop, right_crop = max(0, left_crop), max(0, right_crop)
        if left_crop + right_crop >= image.shape[1]:
            raise ValueError("Invalid crop values: Exceeds image width")

        return image[:, left_crop:image.shape[1] - right_crop]

