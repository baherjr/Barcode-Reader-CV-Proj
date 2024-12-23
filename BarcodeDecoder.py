import numpy as np
import cv2


class BarcodeDecoder:
    TOLERANCE = 1  # Define tolerance value
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
    def within_tolerance(width, target_size):

        return abs(width - target_size) <= BarcodeDecoder.TOLERANCE

    def crack_the_code(self, img):

        mean = img.mean(axis=0)

        mean[mean <= 127] = 1  # Black
        mean[mean > 127] = 0  # White

        pixels = ''.join(mean.astype(np.uint8).astype(str))

        bar_widths = []
        count = 1
        color = pixels[0]

        for i in range(1, len(pixels)):
            if pixels[i] == pixels[i - 1]:
                count += 1
            else:
                bar_widths.append([int(color), count])
                color = pixels[i]
                count = 1

        bar_widths.append([int(color), count])

        black_narrow_bar_size = min(bar[1] for bar in bar_widths if bar[0] == 1)
        black_wide_bar_size = max(bar[1] for bar in bar_widths if bar[0] == 1)
        white_narrow_bar_size = min(bar[1] for bar in bar_widths if bar[0] == 0)
        white_wide_bar_size = max(bar[1] for bar in bar_widths if bar[0] == 0)

        # Decode  barcode
        digits = []
        pixel_index = 0
        current_digit_widths = ""
        skip_next = False

        while pixel_index < len(pixels):
            if skip_next:
                if pixels[pixel_index] == '1':
                    pixel_index += black_narrow_bar_size
                else:
                    pixel_index += white_narrow_bar_size
                skip_next = False
                continue

            count = 1
            try:
                while pixels[pixel_index] == pixels[pixel_index + 1]:
                    count += 1
                    pixel_index += 1
            except IndexError:
                pass

            pixel_index += 1
            current_color = 1 if pixels[pixel_index - 1] == '1' else 0

            if current_color == 1:  # Black bar
                if self.within_tolerance(count, black_narrow_bar_size):
                    current_digit_widths += self.NARROW
                elif self.within_tolerance(count, black_wide_bar_size):
                    current_digit_widths += self.WIDE

            else:  # White bar
                if self.within_tolerance(count, white_narrow_bar_size):
                    current_digit_widths += self.NARROW
                elif self.within_tolerance(count, white_wide_bar_size):
                    current_digit_widths += self.WIDE

            if current_digit_widths in self.CODE11_WIDTHS:
                digits.append(self.CODE11_WIDTHS[current_digit_widths])  # Decode the digit
                current_digit_widths = ""
                skip_next = True

        return digits
