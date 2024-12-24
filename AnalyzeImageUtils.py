import cv2
import numpy as np
from scipy import fftpack


class AnalyzeImage:
    @staticmethod
    def apply_high_pass_filter(fft_shift, cutoff=30):
        h, w = fft_shift.shape
        mask = np.ones((h, w), dtype=bool)

        center_y, center_x = h // 2, w // 2
        y, x = np.ogrid[-center_y:h - center_y, -center_x:w - center_x]
        mask = (x ** 2 + y ** 2 > cutoff ** 2)

        return fft_shift * mask

    @staticmethod
    def calculate_histogram(image):
        histogram = cv2.calcHist([image], [0], None, [256], [0, 256])
        return histogram / histogram.sum()

    @staticmethod
    def fft_and_magnitude_spectrum(image):
        fft_result = np.fft.fft2(image)
        shifted_fft = np.fft.fftshift(fft_result)
        magnitude = 20 * np.log(np.abs(shifted_fft))
        return shifted_fft, magnitude

    def is_this_image_wearing_glasses(image):
        transformed_image, spectrum = AnalyzeImage.fft_and_magnitude_spectrum(image)

        height, width = image.shape
        mid_height, mid_width = height // 2, width // 2
        roi_dim = 9

        region_of_interest = spectrum[
                             mid_height - roi_dim: mid_height + roi_dim + 1,
                             mid_width - roi_dim: mid_width + roi_dim + 1
                             ]
        roi_avg = round(np.mean(region_of_interest))

        corner_regions = [
            spectrum[0:10, 0:10],
            spectrum[0:10, -10:],
            spectrum[-10:, 0:10],
            spectrum[-10:, -10:]
        ]
        corner_avg = sum(round(np.mean(corner)) for corner in corner_regions) / 4

        if np.mean(spectrum[100:-100, 100:-100]) == float('-inf'):
            return "FalseINF"

        low_threshold, high_threshold = 235, 90
        return roi_avg > low_threshold and corner_avg < high_threshold

    def check_if_its_sunbathing(image, light_threshold=220, pixel_ratio=0.905):
        histogram = AnalyzeImage.calculate_histogram(image)
        light_pixels = histogram[light_threshold:].sum()
        total_pixels = np.sum(histogram)
        return light_pixels / total_pixels > pixel_ratio

    @staticmethod
    def is_this_a_midnight_snack(image, dark_threshold=20, pixel_ratio=0.905):
        histogram = AnalyzeImage.calculate_histogram(image)
        dark_pixels = histogram[:dark_threshold].sum()
        total_pixels = np.sum(histogram)
        return dark_pixels / total_pixels > pixel_ratio

    @staticmethod
    def check_contrast(image):
        pixel_ratio = 0.8
        lower_bound, upper_bound = 100, 200
        histogram = AnalyzeImage.calculate_histogram(image)
        grey_pixel_ratio = histogram[lower_bound:upper_bound].sum()
        return grey_pixel_ratio > pixel_ratio

    @staticmethod
    def check_for_salt_and_pepper(image):
        # Convert image to grayscale if it's not already (assuming image is in [0, 255] range)
        if len(image.shape) == 3:
            image = np.mean(image, axis=2).astype(np.uint8)

        # Flatten the image
        flat_image = image.flatten()

        # Count only the extreme pixels that are not part of large continuous areas
        # This helps differentiate between noise and actual image features

        # Define a small window size for local analysis
        window_size = 3

        # Function to check if a pixel is isolated (noise-like)
        def is_isolated_noise(pixel_value, i, j, image):
            if 0 < i < image.shape[0] - 1 and 0 < j < image.shape[1] - 1:
                window = image[i - 1:i + 2, j - 1:j + 2]
                return np.sum(window == pixel_value) <= 1  # Very isolated
            return False

        # Count isolated salt and pepper pixels
        salt_count = 0
        pepper_count = 0
        for i in range(image.shape[0]):
            for j in range(image.shape[1]):
                if image[i, j] == 255 and is_isolated_noise(255, i, j, image):
                    salt_count += 1
                elif image[i, j] == 0 and is_isolated_noise(0, i, j, image):
                    pepper_count += 1

        # Total pixel count
        total_pixels = image.size

        # Calculate percentage but only for isolated extreme pixels
        noise_percentage = (salt_count + pepper_count) / total_pixels

        # Thresholds for detecting noise, made very conservative
        noise_threshold = 0.0005  # 0.05% of pixels, extremely conservative

        # Return only if there's a significant amount of isolated noise
        return noise_percentage > noise_threshold

    @staticmethod
    def is_rotated(img, rotation_threshold=10):
        img_with_border = cv2.copyMakeBorder(img, 100, 100, 200, 200, cv2.BORDER_CONSTANT, value=[255])

        _, inverted_threshold = cv2.threshold(img_with_border, 60, 255, cv2.THRESH_BINARY_INV)

        contours_found, _ = cv2.findContours(inverted_threshold, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if not contours_found:
            raise ValueError("No contours found.")

        max_contour = max(contours_found, key=cv2.contourArea)

        min_area_rect = cv2.minAreaRect(max_contour)

        rotation_angle = min_area_rect[2]

        return abs(rotation_angle) != rotation_threshold

    @staticmethod
    def detect_periodic(img, cutoff=5, factor=50):
        float_img = np.float32(img)
        fft_img = np.fft.fft2(float_img)
        fft_centered = np.fft.fftshift(fft_img)

        filtered_fft = AnalyzeImage.apply_high_pass_filter(fft_centered, cutoff)

        magnitude = np.abs(filtered_fft)
        norm_magnitude = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX)

        peak = np.max(norm_magnitude)
        mean = np.mean(norm_magnitude)
        std = np.std(norm_magnitude)

        score = (peak - mean) / std if std != 0 else 0
        threshold = factor * std

        return score > threshold

    @staticmethod
    def sobel_operation(image):
        grad_x = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
        grad_abs = np.abs(grad_x)
        grad_scaled = np.uint8(255 * grad_abs / np.max(grad_abs))
        return grad_scaled

    @staticmethod
    def spot_the_obstacle_course(image, threshold_value=50, density_threshold=0.7):
        grad_scaled = AnalyzeImage.sobel_operation(image)

        _, binarized = cv2.threshold(grad_scaled, 60, 255, cv2.THRESH_BINARY)
        binarized = cv2.morphologyEx(binarized, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7)))

        col_sum = np.sum(binarized == 255, axis=0)
        dense_columns = col_sum > density_threshold * np.max(col_sum)
        barcode_found = np.sum(dense_columns) > threshold_value

        is_obstructed = False
        if barcode_found:
            for i in range(len(col_sum)):
                if dense_columns[i] and col_sum[i] < (np.max(col_sum) - 70):
                    is_obstructed = True
                    break

        return is_obstructed