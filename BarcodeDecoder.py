import numpy as np
import cv2


class BarcodeDecoder:
    TOLERANCE = 1  # Allowed variation for determining narrow/wide bars
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

    def __init__(self, debug=False):
        """
        Initialize the decoder with debugging capability.
        """
        self.debug = debug # Used for Testing what went wrong can be turned on view ImageProcessor

    @staticmethod
    def within_tolerance(value, target):
        """
        Check if a value is within the tolerance range of the target.
        """
        return abs(value - target) <= BarcodeDecoder.TOLERANCE

    def crack_the_code(self, img):
        """
        Decode a barcode image to extract valid Code11 digits.

        """
        if self.debug:
            print("=== Barcode Decoding Started ===")

        # Step 1: Calculate column-wise mean and binarize
        column_means = img.mean(axis=0)
        if self.debug:
            print(f"Column-wise mean intensities (truncated): {column_means[:20]} ...")

        binary_array = np.where(column_means <= 127, 1, 0)
        binary_pixels = ''.join(binary_array.astype(str))

        if self.debug:
            print(f"Binary representation (truncated): {binary_pixels[:50]} ...")

        # Step 2: Determine bar widths and colors
        bar_widths = self._calculate_bar_widths(binary_pixels)

        # Step 3: Calculate narrow and wide bar sizes
        bar_sizes = self._calculate_bar_sizes(bar_widths)
        black_narrow, black_wide, white_narrow, white_wide = bar_sizes

        if self.debug:
            print(f"Black - Narrow: {black_narrow}, Wide: {black_wide}")
            print(f"White - Narrow: {white_narrow}, Wide: {white_wide}")

        # Step 4: Decode the barcode using measured widths
        digits = self._decode_pixels(binary_pixels, bar_widths, *bar_sizes)

        if self.debug:
            print("\n=== Barcode Decoding Complete ===")
            print(f"Decoded Digits: {digits}")

        return digits

    def _calculate_bar_widths(self, pixels):
        """
        Analyze the binary pixel string to group bars by color and width.
        """
        bar_widths = []
        prev_pixel = pixels[0]
        count = 1

        for i in range(1, len(pixels)):
            if pixels[i] == prev_pixel:
                count += 1
            else:
                bar_widths.append((int(prev_pixel), count))
                prev_pixel = pixels[i]
                count = 1
        bar_widths.append((int(prev_pixel), count))  # Add the last bar

        if self.debug:
            print(f"Bar Widths (truncated): {bar_widths[:10]} ...")
        return bar_widths

    def _calculate_bar_sizes(self, bar_widths):
        """
        Calculate the narrow and wide sizes for both black and white bars.

        """
        black_bars = [width for color, width in bar_widths if color == 1]
        white_bars = [width for color, width in bar_widths if color == 0]

        return (
            min(black_bars), max(black_bars),
            min(white_bars), max(white_bars)
        )

    def _decode_pixels(self, pixels, bar_widths, black_narrow, black_wide, white_narrow, white_wide):
        """
        Decode binary pixels into Code11 digits using the given bar sizes.

        """
        digits = []
        current_digit_widths = ''
        pixel_index = 0
        skip_next = False

        while pixel_index < len(pixels):
            if skip_next:
                pixel_index += white_narrow if pixels[pixel_index] == '0' else black_narrow
                skip_next = False
                continue

            count = 1
            try:
                while pixels[pixel_index] == pixels[pixel_index + 1]:
                    count += 1
                    pixel_index += 1
            except IndexError:
                pass  # End of pixel sequence

            pixel_index += 1  # Move to next pixel
            current_color = 1 if pixels[pixel_index - 1] == '1' else 0
            is_black = current_color == 1

            if self.within_tolerance(count, black_narrow if is_black else white_narrow):
                current_digit_widths += self.NARROW
            elif self.within_tolerance(count, black_wide if is_black else white_wide):
                current_digit_widths += self.WIDE

            if self.debug:
                print(f"Measured bar - Count: {count}, Color: {'Black' if is_black else 'White'}")
                print(f"Current Digit Widths: {current_digit_widths}")

            if current_digit_widths in self.CODE11_WIDTHS:
                digit = self.CODE11_WIDTHS[current_digit_widths]
                digits.append(digit)

                if self.debug:
                    print(f"Decoded Pattern: {current_digit_widths} -> {digit}")

                current_digit_widths = ''
                skip_next = True  # Skip separator on the next iteration

        return digits

