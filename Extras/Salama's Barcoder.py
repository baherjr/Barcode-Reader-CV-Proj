import numpy as np
import cv2  # OpenCV for loading and processing the image

# 0 means narrow, 1 means wide
NARROW = "0"
WIDE = "1"

code11_widths = {
    "00110": "Stop/Start",
    "10001": "1",
    "01001": "2",
    "11000": "3",
    "00101": "4",
    "10100": "5",
    "01100": "6",
    "00011": "7",
    "10010": "8",
    "10000": "9",
    "00001": "0",
    "00100": "-",
}

# Load the image as a grayscale array
your_cropped_image_path = r'../processed_barcodes/06 - meen fata7 el nour 333eenaaayy.jpg'
your_cropped_image = cv2.imread(your_cropped_image_path, cv2.IMREAD_GRAYSCALE)

# Ensure the image was loaded successfully
if your_cropped_image is None:
    raise FileNotFoundError(f"Image not found at {your_cropped_image_path}")

# Get the average of each column in your image
mean = your_cropped_image.mean(axis=0)

# Set it to black or white based on its value
binary = np.zeros_like(mean, dtype=np.uint8)
binary[mean <= 127] = 1
binary[mean > 128] = 0

# Convert to string of pixels in order to loop over it
pixels = ''.join(binary.astype(str))

# Need to figure out how many pixels represent a narrow bar
narrow_bar_size = 0
for pixel in pixels:
    if pixel == "1":
        narrow_bar_size += 1
    else:
        break

wide_bar_size = narrow_bar_size * 2

digits = []
pixel_index = 0
current_digit_widths = ""
skip_next = False

while pixel_index < len(pixels):

    if skip_next:
        pixel_index += narrow_bar_size
        skip_next = False
        continue

    count = 1
    try:
        while pixels[pixel_index] == pixels[pixel_index + 1]:
            count += 1
            pixel_index += 1
    except IndexError:
        break  # Prevent index out of bounds error
    pixel_index += 1

    current_digit_widths += NARROW if count == narrow_bar_size else WIDE

    if current_digit_widths in code11_widths:
        digits.append(code11_widths[current_digit_widths])
        current_digit_widths = ""
        skip_next = True  # Next iteration will be a separator, so skip it

print(digits)