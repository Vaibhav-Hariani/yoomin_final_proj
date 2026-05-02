#!/usr/bin/env python3
"""Render an image onto the LED matrix using calibration_result.py.

The physical display is landscape (64 wide × 40 tall). The LED map uses
standard coordinates (row=0..39 is vertical, col=0..63 is horizontal).
OpenCV loads images as BGR; NeoPixels (WS2812B) expect GRB — channels
are reordered on write.
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

Y_LEDS = 40   # 5 blocks × 8 rows  — vertical (height)
X_LEDS = 64   # 2 blocks × 32 cols — horizontal (width)


def render(image_path: str, preview_path: str = "preview.png"):
    led_map = build_mapping(GRID_ORDER, BLOCK_ORIENTATION)

    img = cv2.imread(image_path)   # BGR
    if img is None:
        raise FileNotFoundError(f"Could not load image: {image_path}")

    # Resize to landscape (64 wide × 40 tall) — cv2.resize takes (width, height)
    frame = cv2.resize(img, (X_LEDS, Y_LEDS), interpolation=cv2.INTER_AREA)

    cv2.imwrite(preview_path, frame)
    print(f"Preview saved to {preview_path}  ({frame.shape[1]}×{frame.shape[0]})")

    # Write to LEDs — convert BGR (OpenCV) → RGB (neopixel library handles GRB internally)
    for y in range(Y_LEDS):
        for x in range(X_LEDS):
            led_index = int(led_map[y, x])
            LEDS[led_index] = (int(frame[y, x, 2]),   # R
                               int(frame[y, x, 1]),   # G
                               int(frame[y, x, 0]))   # B
    LEDS.show()
    print("Frame written to LEDs.")


if __name__ == "__main__":
    image_path = sys.argv[1] if len(sys.argv) > 1 else "image_test.jpeg"
    render(image_path)
