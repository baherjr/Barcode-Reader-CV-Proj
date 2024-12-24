import numpy as np
import cv2
from scipy import fftpack


class ImageFrequencyTransformer:

    @staticmethod
    def perform_fourier_transform(image):
        """
        Performs Fourier Transform and shifts the zero frequency component to the center.
        """
        fourier_transformed = fftpack.fft2(image)
        shifted_transform = fftpack.fftshift(fourier_transformed)
        return fourier_transformed, shifted_transform

    @staticmethod
    def create_noise_mask(shifted_transform, image_shape, magnitude_threshold):
        """
        Creates a mask to suppress noise based on the magnitude spectrum.
        """
        magnitude_spectrum = np.abs(shifted_transform)
        max_magnitude_value = np.max(magnitude_spectrum)

        rows, cols = image_shape
        noise_cutoff = max_magnitude_value * magnitude_threshold

        noise_mask = np.ones((rows, cols), dtype=np.float32)
        noise_mask[magnitude_spectrum >= noise_cutoff] = 0

        return noise_mask

    @staticmethod
    def apply_mask_and_inverse_transform(shifted_transform, noise_mask):
        """
        Applies the noise mask and performs inverse Fourier Transform to return to the spatial domain.
        """
        filtered_transform_shifted = shifted_transform * noise_mask
        filtered_transform = fftpack.ifftshift(filtered_transform_shifted)
        filtered_image = fftpack.ifft2(filtered_transform).real
        return filtered_image

    @staticmethod
    def normalize_and_threshold(filtered_image):
        """
        Normalizes the filtered image and applies thresholding.
        """
        normalized_image = cv2.normalize(filtered_image, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        _, thresholded_image = cv2.threshold(normalized_image, 105, 255, cv2.THRESH_BINARY)
        return thresholded_image

    @staticmethod
    def remove_periodic_noise(image, magnitude_threshold=None):
        """
        Removes periodic noise from an image.
        The magnitude threshold is dynamically determined if not provided.
        """
        # Step 1: Perform Fourier Transform
        fourier_transformed, shifted_transform = (
            ImageFrequencyTransformer.perform_fourier_transform(image)
        )

        if magnitude_threshold is None:
            # Calculate the magnitude spectrum
            magnitude_spectrum = np.abs(shifted_transform)
            # Use a fraction of global average magnitude as the dynamic threshold
            average_magnitude = np.mean(magnitude_spectrum)
            magnitude_threshold = average_magnitude * 0.05  # Experimentally chosen factor

        # Step 3: Calculate Noise Mask
        noise_mask = ImageFrequencyTransformer.create_noise_mask(
            shifted_transform, image.shape, magnitude_threshold
        )

        # Step 4: Apply Mask in Frequency Domain
        filtered_image = ImageFrequencyTransformer.apply_mask_and_inverse_transform(
            shifted_transform, noise_mask
        )

        # Step 5: Normalize and Threshold
        final_image = ImageFrequencyTransformer.normalize_and_threshold(filtered_image)

        return final_image

    def apply_sharpen_filter(image, sharpening_strength=None):
        """
        Applies a sharpening filter to enhance fine details in the image.
        The sharpening strength is dynamically determined if not provided.
        """
        # Define a Laplacian kernel for sharpening
        sharpen_kernel = np.array([[-1, -1, -1],
                                   [-1, 9, -1],
                                   [-1, -1, -1]])

        # Dynamically determine sharpening strength if not provided
        if sharpening_strength is None:
            # Calculate sharpness score using a Laplacian variance measure
            sharpness_score = cv2.Laplacian(image, cv2.CV_64F).var()

            # Define ranges for sharpening strength based on sharpness score
            if sharpness_score < 50:  # Low sharpness
                sharpening_strength = 0.6
            elif 50 <= sharpness_score < 100:  # Moderate sharpness
                sharpening_strength = 0.3
            else:  # High sharpness
                sharpening_strength = 0.1

        # Apply the Laplacian kernel to the image to emphasize details
        laplacian_filtered = cv2.filter2D(image, -1, sharpen_kernel)

        # Combine the original and filtered images with adjustable strength
        sharpened_image = cv2.addWeighted(image, 1 + sharpening_strength, laplacian_filtered, -sharpening_strength, 0)

        # Apply thresholding to emphasize edges
        _, thresholded_image = cv2.threshold(sharpened_image, 150, 255, cv2.THRESH_BINARY)
        return thresholded_image
