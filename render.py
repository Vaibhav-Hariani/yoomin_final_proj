#!/usr/bin/env python3
"""Render an image onto the LED matrix using calibration_result.py.

The physical display is portrait (40 wide × 64 tall). The LED map uses a
transposed coordinate system (row=0..39 is horizontal, col=0..63 is vertical),
so the image is resized to portrait then transposed before writing.

LEDs expect BGR byte order, which matches OpenCV's native channel layout —
no channel swapping is needed.
"""

import sys
import numpy as np
import cv2

try:
    from calibration_result import GRID_ORDER, BLOCK_ORIENTATION
except ImportError:
    raise SystemExit("calibration_result.py not found — run calibrate.py first")

from mapping import build_mapping
from writer import LEDS

Y_LEDS = 40   # 5 blocks × 8 rows  (horizontal axis on physical display)
X_LEDS = 64   # 2 blocks × 32 cols (vertical axis on physical display)


def render(image_path: str, preview_path: str = "preview.png"):
    led_map = build_mapping(GRID_ORDER, BLOCK_ORIENTATION)

    img = cv2.imread(image_path)   # BGR
    if img is None:
        raise FileNotFoundError(f"Could not load image: {image_path}")

    # Resize to portrait (40 wide × 64 tall) — cv2.resize takes (width, height)
    portrait = cv2.resize(img, (Y_LEDS, X_LEDS), interpolation=cv2.INTER_AREA)

    # Preview is the portrait image — exactly what the display shows
    cv2.imwrite(preview_path, portrait)
    print(f"Preview saved to {preview_path}  ({portrait.shape[1]}×{portrait.shape[0]})")

    # Transpose to (40, 64, 3) so frame[y, x] aligns with led_map[y, x]
    frame = np.transpose(portrait, (1, 0, 2))

    # Write to LEDs — frame is BGR (OpenCV), LEDs expect BGR: direct passthrough
    for y in range(Y_LEDS):
        for x in range(X_LEDS):
            led_index = int(led_map[y, x])
            LEDS[led_index] = (int(frame[y, x, 0]),   # B
                               int(frame[y, x, 1]),   # G
                               int(frame[y, x, 2]))   # R
    LEDS.show()
    print("Frame written to LEDs.")


if __name__ == "__main__":
    image_path = sys.argv[1] if len(sys.argv) > 1 else "image_test.jpeg"
    render(image_path)
