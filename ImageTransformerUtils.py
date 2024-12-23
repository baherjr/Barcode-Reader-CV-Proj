import numpy as np
import cv2
from scipy import fftpack


class ImageTransformer:

    @staticmethod
    def periodic_noise_removal(img, threshold=0.1):

        fft_image = fftpack.fft2(img)
        fft_shifted = fftpack.fftshift(fft_image)

        magnitude_spectrum = np.abs(fft_shifted)
        max_magnitude = np.max(magnitude_spectrum)

        rows, cols = img.shape
        crow, ccol = rows // 2, cols // 2

        mask = np.ones((rows, cols), dtype=np.float32)
        cutoff_magnitude = max_magnitude * threshold

        mask[magnitude_spectrum >= cutoff_magnitude] = 0
        f_transform_shifted_filtered = fft_shifted * mask

        f_transform_filtered = fftpack.ifftshift(f_transform_shifted_filtered)
        filtered_image = fftpack.ifft2(f_transform_filtered).real

        filtered_image = cv2.normalize(filtered_image, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        _, threshed = cv2.threshold(filtered_image, 105, 255, cv2.THRESH_BINARY)
        return threshed

    @staticmethod
    def Sharpen(image):

        strength = 0.3
        laplacian_kernel = np.array([[0, -1, 0],
                                     [-1, 5, -1],
                                     [0, -1, 0]])

        laplacian_image = cv2.filter2D(image, -1, laplacian_kernel)
        sharpened_image = cv2.addWeighted(image, 1 + strength, laplacian_image, -strength, 0)
        _, threshed = cv2.threshold(sharpened_image, 150, 255, cv2.THRESH_BINARY)
        return threshed

