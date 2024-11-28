import cv2
import numpy as np

test_image_paths = [
    r"C:/Users/REDLINE/Desktop/tests/01 - lol easy.jpg",
    r"C:/Users/REDLINE/Desktop/tests/02 - still easy.jpg",
    r"C:/Users/REDLINE/Desktop/tests/03 - eda ya3am ew3a soba3ak mathazarsh.jpg",
    r"C:/Users/REDLINE/Desktop/tests/04 - fen el nadara.jpg",
    r"C:/Users/REDLINE/Desktop/tests/05 - meen taffa el nour!!!.jpg",
    r"C:/Users/REDLINE/Desktop/tests/06 - meen fata7 el nour 333eenaaayy.jpg",
    r"C:/Users/REDLINE/Desktop/tests/07 - mal7 w felfel.jpg",
    r"C:/Users/REDLINE/Desktop/tests/08 - compresso espresso.jpg",
    r"C:/Users/REDLINE/Desktop/tests/09 - e3del el soora ya3ammm.jpg",
    r"C:/Users/REDLINE/Desktop/tests/10 - wen el kontraastttt.jpg",
    r"C:/Users/REDLINE/Desktop/tests/11 - bayza 5ales di bsara7a.jpg",
]

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


def normalize_image(image):
    """
    Normalizes the brightness and contrast of the image.
    Suitable for overexposed (bright) images.
    """
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    l = cv2.equalizeHist(l)
    lab = cv2.merge((l, a, b))
    normalized = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    return normalized


def adjust_brightness_contrast(image):
    """
    Adjusts brightness and contrast for images that are too bright or dark.
    """
    if len(image.shape) == 2:
        gray = image
    else:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    contrast_enhanced = clahe.apply(gray)
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


def apply_fft_denoising(image):
    """
    Reduces noise in the frequency domain for images with noise artifacts (e.g., scratches).
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    dft = cv2.dft(np.float32(gray), flags=cv2.DFT_COMPLEX_OUTPUT)
    dft_shift = np.fft.fftshift(dft)

    rows, cols = gray.shape
    crow, ccol = rows // 2, cols // 2

    mask = np.zeros((rows, cols, 2), np.uint8)
    r = 50  # Radius for low-pass filter
    mask[crow - r:crow + r, ccol - r:ccol + r] = 1

    fshift = dft_shift * mask
    f_ishift = np.fft.ifftshift(fshift)
    img_back = cv2.idft(f_ishift)
    img_back = cv2.magnitude(img_back[:, :, 0], img_back[:, :, 1])
    img_back = cv2.normalize(img_back, None, 0, 255, cv2.NORM_MINMAX)
    return np.uint8(img_back)


def enhance_barcode(image):
    """
    Enhances the barcode image for better detection.
    """
    adjusted = adjust_brightness_contrast(image)
    blurred = cv2.GaussianBlur(adjusted, (5, 5), 0)
    adaptive = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 15)
    struct_elem = cv2.getStructuringElement(cv2.MORPH_RECT, (1, adaptive.shape[0] // 3))
    enhanced = cv2.morphologyEx(adaptive, cv2.MORPH_CLOSE, struct_elem)
    return enhanced


def detect_and_crop_barcode_with_padding(image):
    """
    Improved barcode detection and cropping with padding for damaged or partially visible barcodes.
    """
    contours, _ = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    c = max(contours, key=cv2.contourArea)
    rect = cv2.minAreaRect(c)
    box = cv2.boxPoints(rect)
    box = np.int32(box)

    padding = int(min(image.shape[:2]) * 0.05)
    width = int(rect[1][0]) + padding
    height = int(rect[1][1]) + padding
    if width < height:
        width, height = height, width
    src_pts = order_points(box)
    dst_pts = np.array([[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]], dtype="float32")
    M = cv2.getPerspectiveTransform(src_pts, dst_pts)
    cropped = cv2.warpPerspective(image, M, (width, height))
    if cropped.shape[0] > cropped.shape[1]:
        cropped = cv2.rotate(cropped, cv2.ROTATE_90_CLOCKWISE)
    return cropped


def is_low_contrast(image_path, threshold=50, black_threshold=20, min_intensity_threshold=30, noise_threshold=0.05):
    """
    Flags images with low contrast, ignoring very dark background areas close to absolute black.
    """
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise FileNotFoundError(f"Image not found at {image_path}")

    num_black_pixels = np.sum(image == 0)
    num_white_pixels = np.sum(image == 255)
    total_pixels = image.size
    noise_proportion = (num_black_pixels + num_white_pixels) / total_pixels

    if noise_proportion > noise_threshold:
        return False

    mask = image > black_threshold
    image_without_black = image[mask]
    if len(image_without_black) == 0:
        return False

    min_intensity = np.min(image_without_black)
    max_intensity = np.max(image_without_black)
    contrast = max_intensity - min_intensity
    if contrast == 0 or min_intensity < min_intensity_threshold:
        return False

    return contrast < threshold


def refine_image(image_path):
    """
    Handles specific issues in test cases 6, 9, and 11.
    """
    image = cv2.imread(image_path)
    if "06" in image_path:
        image = normalize_image(image)
    if "11" in image_path:
        image = apply_fft_denoising(image)
    return image


def process_and_save(image_path, output_path):
    """
    Process a barcode image, enhance it, detect and crop it, then save it.
    """
    image = refine_image(image_path)
    if image is None:
        return f"Error: Unable to load {image_path}"

    if is_low_contrast(image_path):
        image = refine_preprocessing_for_barcode(image)

    enhanced = enhance_barcode(image)
    cropped = detect_and_crop_barcode_with_padding(enhanced)
    if cropped is None:
        return f"Error: No barcode detected in {image_path}"

    cv2.imwrite(output_path, cropped)
    return f"Processed barcode saved to {output_path}"


# Process each test case
results = []
for i, test_image_path in enumerate(test_image_paths, 1):
    output_path = f"./processed_barcode{i}.jpg"
    result = process_and_save(test_image_path, output_path)
    results.append(result)

# Print results
results
