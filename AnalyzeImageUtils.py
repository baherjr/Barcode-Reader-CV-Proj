import cv2
import numpy as np
from scipy import fftpack
from PreprocessImageUtils import PreprocessImage


class AnalyzeImage:
    @staticmethod
    def high_pass_filter(fft_shift, cutoff_frequency=30):
        rows, cols = fft_shift.shape
        mask = np.ones((rows, cols))

        center_row, center_col = rows // 2, cols // 2
        mask[center_row - cutoff_frequency:center_row + cutoff_frequency,
        center_col - cutoff_frequency:center_col + cutoff_frequency] = 0

        fft_shift_hp = fft_shift * mask
        return fft_shift_hp

    @staticmethod
    def detect_blurred(image):
        f_transform = np.fft.fft2(image)
        f_transform_shifted = np.fft.fftshift(f_transform)
        magnitude_spectrum = 20 * np.log(np.abs(f_transform_shifted))

        rows, cols = image.shape
        center_row = rows // 2
        center_col = cols // 2
        roi_size = 9
        roi = magnitude_spectrum[center_row - roi_size:center_row + roi_size + 1,
              center_col - roi_size:center_col + roi_size + 1]
        avg_roi = round(np.mean(roi))
        corner1_avg = round(np.mean(magnitude_spectrum[0:10, 0:10]))
        corner2_avg = round(np.mean(magnitude_spectrum[0:10, -10:]))
        corner3_avg = round(np.mean(magnitude_spectrum[-10:, 0:10]))
        corner4_avg = round(np.mean(magnitude_spectrum[-10:, -10:]))
        corners_avg = (corner1_avg + corner2_avg + corner3_avg + corner4_avg) / 4

        if np.mean(magnitude_spectrum[100:-100, 100:-100]) == float('-inf'):
            return "FalseINF"

        low_avg_threshold = 235
        high_avg_threshold = 90

        if avg_roi > low_avg_threshold and corners_avg < high_avg_threshold:
            return True
        else:
            return False

    @staticmethod
    def detect_high_brightness(image):
        pixel_ratio = 0.905
        dark_threshold = 20
        light_threshold = 220

        histogram = cv2.calcHist([image], [0], None, [256], [0, 256])
        histogram = histogram / histogram.sum()
        light_pixels_ratio = histogram[light_threshold:].sum()
        dark_pixels_ratio = histogram[:dark_threshold].sum()

        if light_pixels_ratio > pixel_ratio:
            return True
        else:
            return False

    @staticmethod
    def detect_low_brightness(image):
        pixel_ratio = 0.905
        dark_threshold = 20
        light_threshold = 220

        histogram = cv2.calcHist([image], [0], None, [256], [0, 256])
        histogram = histogram / histogram.sum()
        light_pixels_ratio = histogram[light_threshold:].sum()
        dark_pixels_ratio = histogram[:dark_threshold].sum()
        if dark_pixels_ratio > pixel_ratio:
            return True
        else:
            return False

    @staticmethod
    def detect_contrast(image):
        pixel_ratio = 0.8
        lower_bound = 100
        upper_bound = 200

        histogram = cv2.calcHist([image], [0], None, [256], [0, 256])
        histogram = histogram / histogram.sum()

        grey_pixel_ratio = histogram[lower_bound:upper_bound].sum()
        if grey_pixel_ratio > pixel_ratio:
            return True
        else:
            return False

    @staticmethod
    def detect_sltnp(image):
        f_transform = np.fft.fft2(image)
        f_transform_shifted = np.fft.fftshift(f_transform)
        magnitude_spectrum = 20 * np.log(np.abs(f_transform_shifted))

        rows, cols = image.shape
        center_row = rows // 2
        center_col = cols // 2
        roi_size = 9
        dc_component = magnitude_spectrum[center_row, center_col]
        roi = magnitude_spectrum[center_row - roi_size:center_row + roi_size + 1,
              center_col - roi_size:center_col + roi_size + 1]
        avg_roi = round(np.mean(roi))
        corner1_avg = round(np.mean(magnitude_spectrum[0:10, 0:10]))
        corner2_avg = round(np.mean(magnitude_spectrum[0:10, -10:]))
        corner3_avg = round(np.mean(magnitude_spectrum[-10:, 0:10]))
        corner4_avg = round(np.mean(magnitude_spectrum[-10:, -10:]))
        corners_avg = (corner1_avg + corner2_avg + corner3_avg + corner4_avg) / 4

        if np.mean(magnitude_spectrum[100:-100, 100:-100]) == float('-inf'):
            return "FalseINF"

        low_avg_threshold = 200
        high_avg_threshold = 200
        if avg_roi > low_avg_threshold and corners_avg > high_avg_threshold:
            return True
        else:
            return False

    @staticmethod
    def is_rotated(img, rotation_threshold=10):
        padded_image = cv2.copyMakeBorder(
            img, 100, 100, 200, 200, cv2.BORDER_CONSTANT, value=[255]
        )

        _, thresh = cv2.threshold(padded_image, 60, 255, cv2.THRESH_BINARY_INV)

        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if not contours:
            raise ValueError("No contours found in the image.")

        largest_contour = max(contours, key=cv2.contourArea)

        rect = cv2.minAreaRect(largest_contour)

        # Extract the rotation angle from the rectangle
        angle = rect[2]
        print(angle)
        # Check if the rotation exceeds the threshold
        is_rotated = (abs(angle) != rotation_threshold)

        return is_rotated

    @staticmethod
    def detect_periodic(image):
        # Convert image to float for better precision in FFT operations
        image_float = image.astype(np.float32)

        # Perform 2D FFT
        fft = np.fft.fft2(image_float)
        fft_shift = np.fft.fftshift(fft)

        # Apply high-pass filter to remove low-frequency components
        fft_shift = AnalyzeImage.high_pass_filter(fft_shift, cutoff_frequency=5)

        # Calculate magnitude spectrum
        magnitude_spectrum = np.abs(fft_shift)

        # Normalize the magnitude spectrum for visualization and analysis
        magnitude_spectrum_normalized = cv2.normalize(magnitude_spectrum, None, 0, 255, cv2.NORM_MINMAX)

        # Find the maximum value and its position in the spectrum
        max_value = np.max(magnitude_spectrum_normalized)
        max_position = np.unravel_index(np.argmax(magnitude_spectrum_normalized), magnitude_spectrum_normalized.shape)

        # Calculate mean and standard deviation for statistical analysis
        mean_value = np.mean(magnitude_spectrum_normalized)
        std_deviation = np.std(magnitude_spectrum_normalized)

        # Compute z-score of the maximum value
        z_score = (max_value - mean_value) / std_deviation if std_deviation != 0 else 0

        print(f"Magnitude Spectrum (Normalized):")
        print(magnitude_spectrum_normalized[:5, :5])  # Print first 5x5 segment for brevity
        print(f"Highest Value in Magnitude Spectrum: {max_value}")
        print(f"Position of Highest Value: {max_position}")
        print(f"Standard Deviation: {std_deviation}")
        print(f"Z-score of the Maximum Value: {z_score}")

        # Detection threshold based on z-score
        detection_threshold = 50 * std_deviation  # Adjust this value if needed
        is_periodic = z_score > detection_threshold

        return is_periodic

    @staticmethod
    def obstacle_detection(image):
        sobel = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
        sobel_abs = np.absolute(sobel)
        sobel_normalized = np.uint8(255 * sobel_abs / np.max(sobel_abs))
        _, binary = cv2.threshold(sobel_normalized, 60, 255, cv2.THRESH_BINARY)
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7)))
        vertical_sum = np.sum(binary == 255, axis=0)
        dense_columns = vertical_sum > 0.7 * np.max(vertical_sum)
        barcode_detected = np.sum(dense_columns) > 50
        is_obstructed = False
        if barcode_detected:
            for i in range(len(vertical_sum)):
                if dense_columns[i] and vertical_sum[i] < (np.max(vertical_sum) - 70):
                    is_obstructed = True
                    break

        return is_obstructed

