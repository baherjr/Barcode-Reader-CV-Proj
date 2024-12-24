import numpy as np
import cv2


class BarcodeDecoder:
    NARROW = "0"
    WIDE = "1"

    # Code11 barcode patterns
    CODE11_WIDTHS = {
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

    @staticmethod
    def determine_bar_sizes(self, pixels):
        """
        Analyzes the bar widths in the barcode and determines narrow and wide sizes.
        """
        # Measure the width of consecutive bars
        bar_widths = []
        count = 1

        for i in range(1, len(pixels)):
            if pixels[i] == pixels[i - 1]:
                count += 1
            else:
                bar_widths.append(count)
                count = 1
        bar_widths.append(count)  # Add last bar width

        narrow = min(bar_widths)  # Narrow is the smallest bar width
        wide_candidates = [width for width in set(bar_widths) if width > narrow]
        wide = min(wide_candidates, default=narrow * 2)  # Default to 2x narrow if no candidates found

        return narrow, wide
    @staticmethod
    def crack_the_code(self, img):
        """
        Decodes a given processed image array of a barcode and returns the result as a list of digits.
        :param img: Processed barcode image (2D array, grayscale or binary).
        :return: Decoded digits as a list.
        """
        # Get the average intensity of each column in the image (binarized input)
        mean = img.mean(axis=0)

        # Convert to binary representation
        binary = np.zeros_like(mean, dtype=np.uint8)
        binary[mean <= 127] = 1  # Modify threshold if needed
        binary[mean > 128] = 0  # Treat everything else as background

        # Convert binary array into a string of pixels
        pixels = ''.join(binary.astype(str))

        # Dynamically determine narrow and wide sizes
        narrow, wide = self.determine_bar_sizes(pixels)

        # Decode the pixel sequence into digits
        digits = []
        pixel_index = 0
        current_digit_widths = ""
        skip_next = False

        while pixel_index < len(pixels):
            if skip_next:
                # Skip the separator
                pixel_index += narrow
                skip_next = False
                continue

            # Measure the width of consecutive pixels
            count = 1
            try:
                while pixels[pixel_index] == pixels[pixel_index + 1]:
                    count += 1
                    pixel_index += 1
            except IndexError:
                break  # Exit if we reach the end of the pixels string

            pixel_index += 1

            # Classify the bar as narrow or wide based on its width
            if narrow - 1 <= count <= narrow + 1:  # Allow for tolerance
                current_digit_widths += self.NARROW
            elif wide - 2 <= count <= wide + 2:  # Use a range for wide classification
                current_digit_widths += self.WIDE
            else:
                # If the width doesn't match, ignore this segment (likely noise)
                continue

            # Check if the extracted widths match a digit pattern
            if current_digit_widths in self.CODE11_WIDTHS:
                digits.append(self.CODE11_WIDTHS[current_digit_widths])
                current_digit_widths = ""  # Reset for the next digit
                skip_next = True  # Skip the separator on the next iteration

        return digits
