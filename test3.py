import cv2
import numpy as np

# Define paths for uploaded images
test_image_1_path = r"Test Cases-20241123/01 - lol easy.jpg"
test_image_2_path = r"Test Cases-20241123/02 - still easy.jpg"
test_image_3_path = r"Test Cases-20241123/03 - eda ya3am ew3a soba3ak mathazarsh.jpg"
test_image_4_path = r"Test Cases-20241123/04 - fen el nadara.jpg"
test_image_5_path = r"Test Cases-20241123/05 - meen taffa el nour!!!.jpg"
test_image_6_path = r"Test Cases-20241123/06 - meen fata7 el nour 333eenaaayy.jpg"
test_image_7_path = r"Test Cases-20241123/07 - mal7 w felfel.jpg"
test_image_8_path = r"Test Cases-20241123/08 - compresso espresso.jpg"
test_image_9_path = r"Test Cases-20241123/09 - e3del el soora ya3ammm.jpg"
test_image_10_path = r"Test Cases-20241123/10 - wen el kontraastttt.jpg"
test_image_11_path = r"Test Cases-20241123/11 - bayza 5ales di bsara7a.jpg"


def order_points(pts):
    """
    Orders points in top-left, top-right, bottom-right, bottom-left order.
    """
    rect = np.zeros((4, 2), dtype=np.float32)
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect


def adjust_brightness_contrast(image):
    """
    Adjusts brightness and contrast for images that are too bright or dark.
    """
    # Check if the image is already in grayscale
    if len(image.shape) == 2:
        gray = image
    else:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Enhance low contrast images (e.g., test 10)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    contrast_enhanced = clahe.apply(gray)

    # Apply gamma correction for very dark images (e.g., test 5)
    gamma_corrected = np.power(contrast_enhanced / float(np.max(contrast_enhanced)), 1.5) * 255
    gamma_corrected = np.uint8(gamma_corrected)

    return gamma_corrected



def refine_preprocessing_for_barcode(img):
    """
    Refines preprocessing for better barcode readability.
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    equalized = cv2.equalizeHist(gray)
    _, binary = cv2.threshold(equalized, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    morph = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=2)
    return morph


def enhance_barcode(image):
    """
    Enhance the barcode image for better detection.
    """
    adjusted = adjust_brightness_contrast(image)
    blurred = cv2.GaussianBlur(adjusted, (5, 5), 0)
    adaptive = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 15)
    struct_elem = cv2.getStructuringElement(cv2.MORPH_RECT, (1, adaptive.shape[0] // 3))
    enhanced = cv2.morphologyEx(adaptive, cv2.MORPH_CLOSE, struct_elem)
    return enhanced


def rotate_image(image, angle):
    """
    Rotates an image by the given angle.
    """
    center = tuple(np.array(image.shape[1::-1]) / 2)
    rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated_image = cv2.warpAffine(image, rotation_matrix, image.shape[1::-1], flags=cv2.INTER_LINEAR)
    return rotated_image


def detect_and_crop_barcode(image):
    """
    Detects and crops a barcode from an enhanced image.
    """
    contours, _ = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    c = max(contours, key=cv2.contourArea)
    rect = cv2.minAreaRect(c)
    box = cv2.boxPoints(rect)
    box = np.int32(box)
    width = int(rect[1][0])
    height = int(rect[1][1])
    if width < height:
        width, height = height, width
    src_pts = order_points(box)
    dst_pts = np.array([[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]], dtype="float32")
    M = cv2.getPerspectiveTransform(src_pts, dst_pts)
    cropped = cv2.warpPerspective(image, M, (width, height))
    if cropped.shape[0] > cropped.shape[1]:
        cropped = cv2.rotate(cropped, cv2.ROTATE_90_CLOCKWISE)
    return cropped

def is_low_contrast(image_path, threshold=50, black_threshold=20, min_intensity_threshold=30):
    """
    Flags images with low contrast, ignoring very dark background areas close to absolute black.
    Skips cases where the image has minimal contrast (e.g., contrast = 0) due to a uniform dark region.

    Args:
        image_path (str): Path to the image.
        threshold (int): Minimum contrast value to flag low contrast images.
        black_threshold (int): Grayscale value considered "close to black" to ignore in contrast calculation.
        min_intensity_threshold (int): Minimum intensity value to ignore images with near-zero contrast.

    Returns:
        bool: True if the image has low contrast, False otherwise.
    """
    # Load the image in grayscale
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    # Check if the image was loaded
    if image is None:
        raise FileNotFoundError(f"Image not found at {image_path}")

    # Create a mask to exclude pixels close to black
    mask = image > black_threshold  # True for pixels that are not near black

    # Mask the image to exclude dark regions
    image_without_black = image[mask]

    # Calculate the pixel intensity range (contrast) only on non-black regions
    if len(image_without_black) == 0:
        return False  # No non-black pixels, so no contrast

    min_intensity = np.min(image_without_black)
    max_intensity = np.max(image_without_black)
    contrast = max_intensity - min_intensity

    # Debugging info
    print(f"Min Intensity (non-black): {min_intensity}, Max Intensity (non-black): {max_intensity}, Contrast: {contrast}")

    # Skip if the contrast is zero or minimal (e.g., the image is almost uniform in intensity)
    if contrast == 0 or min_intensity < min_intensity_threshold:
        print(f"Skipping image due to minimal contrast: Min Intensity {min_intensity}")
        return False

    # Compare against the threshold
    return contrast < threshold


def process_and_save(image_path, output_path):
    """
    Process a barcode image, enhance it, detect and crop it, then save it.
    """
    image = cv2.imread(image_path)
    if image is None:
        return f"Error: Unable to load {image_path}"

    if is_low_contrast(image_path):
        # Apply refinement for better preprocessing
        image = refine_preprocessing_for_barcode(image)

    enhanced = enhance_barcode(image)
    cropped = detect_and_crop_barcode(enhanced)
    if cropped is None:
        return f"Error: No barcode detected in {image_path}"

    cv2.imwrite(output_path, cropped)
    return f"Processed barcode saved to {output_path}"


# Process each test case
result_1 = process_and_save(test_image_1_path, "processed_barcodes/processed_barcode1.jpg")
result_2 = process_and_save(test_image_2_path, "processed_barcodes/processed_barcode2.jpg")
result_3 = process_and_save(test_image_3_path, "processed_barcodes/processed_barcode3.jpg")
result_4 = process_and_save(test_image_4_path, "processed_barcodes/processed_barcode4.jpg")
result_5 = process_and_save(test_image_5_path, "processed_barcodes/processed_barcode5.jpg")
result_6 = process_and_save(test_image_6_path, "processed_barcodes/processed_barcode6.jpg")
result_7 = process_and_save(test_image_7_path, "processed_barcodes/processed_barcode7.jpg")
result_8 = process_and_save(test_image_8_path, "processed_barcodes/processed_barcode8.jpg")
result_9 = process_and_save(test_image_9_path, "processed_barcodes/processed_barcode9.jpg")
result_10 = process_and_save(test_image_10_path, "processed_barcodes/processed_barcode10.jpg")
result_11 = process_and_save(test_image_11_path, "processed_barcodes/processed_barcode11.jpg")

# Return results
result_1, result_2, result_3, result_4, result_5, result_6, result_7, result_8, result_9, result_10, result_11
