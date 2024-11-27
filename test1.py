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
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Enhance low contrast images
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    contrast_enhanced = clahe.apply(gray)

    # Apply gamma correction
    gamma_corrected = np.power(contrast_enhanced / float(np.max(contrast_enhanced)), 1.5) * 255
    gamma_corrected = np.uint8(gamma_corrected)

    return gamma_corrected


def enhance_barcode(image):
    """
    Enhance the barcode image for better detection.
    """
    # Adjust brightness and contrast
    adjusted = adjust_brightness_contrast(image)

    # Apply Gaussian blur to reduce noise
    blurred = cv2.GaussianBlur(adjusted, (5, 5), 0)

    # Adaptive Thresholding for binary image
    adaptive = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 15)

    # Create a larger vertical structuring element for better gap filling
    struct_elem = cv2.getStructuringElement(cv2.MORPH_RECT, (1, adaptive.shape[0] // 3))  # Make it taller

    # Apply closing to fill in gaps
    enhanced = cv2.morphologyEx(adaptive, cv2.MORPH_CLOSE, struct_elem)

    return enhanced


def detect_and_crop_barcode(image):
    """
    Detects and crops a barcode from an input image using a robust approach.
    """
    # Convert to grayscale
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()

    # Denoise the image using Non-Local Means Denoising (good for removing noise)
    denoised = cv2.fastNlMeansDenoising(gray, None, 30, 7, 21)

    # Adaptive thresholding to handle varying brightness
    adaptive_thresh = cv2.adaptiveThreshold(denoised, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)

    # Use Canny edge detection to detect strong edges
    edges = cv2.Canny(adaptive_thresh, threshold1=50, threshold2=150)

    # Perform morphological operations to clean the image and reduce noise
    kernel = np.ones((5, 5), np.uint8)
    dilated = cv2.dilate(edges, kernel, iterations=2)

    # Find contours on the edges image
    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        return None  # No barcode detected

    # Filter contours by area and shape
    valid_contours = []
    for c in contours:
        # Approximate contour to polygon (helps in filtering based on shape)
        epsilon = 0.04 * cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, epsilon, True)

        # Check if the contour has 4 vertices (for quadrilaterals)
        if len(approx) == 4:
            valid_contours.append(approx)

    if not valid_contours:
        return None  # No valid contour found

    # Get the largest valid contour (barcode)
    largest_contour = max(valid_contours, key=cv2.contourArea)

    # Get the bounding box of the largest contour
    rect = cv2.minAreaRect(largest_contour)
    box = cv2.boxPoints(rect)
    box = np.int32(box)

    # Perform perspective transform to crop the barcode region
    width = int(rect[1][0])
    height = int(rect[1][1])
    if width < height:
        width, height = height, width

    src_pts = order_points(box)
    dst_pts = np.array([[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]], dtype="float32")
    M = cv2.getPerspectiveTransform(src_pts, dst_pts)
    cropped = cv2.warpPerspective(gray, M, (width, height))

    return cropped


def process_and_save(image_path, output_path):
    """
    Process a barcode image, enhance it, detect and crop it, then save it.
    """
    image = cv2.imread(image_path)
    if image is None:
        return f"Error: Unable to load {image_path}"

    # Enhance the image first to improve detection
    enhanced = enhance_barcode(image)

    # Detect and crop the barcode
    cropped = detect_and_crop_barcode(enhanced)
    if cropped is None:
        return f"Error: No barcode detected in {image_path}"

    # Save the processed result
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
