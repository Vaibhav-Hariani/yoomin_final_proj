#!/usr/bin/env python3
"""Render an image onto the LED matrix using calibration_result.py.

Downsamples the source image to the LED grid (40×64), writes each pixel to
the correct LED via the calibrated mapping, and saves a scaled-up preview PNG
showing exactly what will appear on the board.

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

Y_LEDS = 40   # 5 blocks × 8 rows
X_LEDS = 64   # 2 blocks × 32 cols
PREVIEW_SCALE = 8   # scale factor for the saved preview PNG


def render(image_path: str, preview_path: str = "preview.png"):
    led_map = build_mapping(GRID_ORDER, BLOCK_ORIENTATION)

    img = cv2.imread(image_path)   # BGR
    if img is None:
        raise FileNotFoundError(f"Could not load image: {image_path}")

    # Downsample to LED grid — INTER_AREA minimises aliasing when shrinking
    frame = cv2.resize(img, (X_LEDS, Y_LEDS), interpolation=cv2.INTER_AREA)

    # Save a scaled-up preview with nearest-neighbour to keep pixels sharp
    preview = cv2.resize(
        frame,
        (X_LEDS * PREVIEW_SCALE, Y_LEDS * PREVIEW_SCALE),
        interpolation=cv2.INTER_NEAREST,
    )
    cv2.imwrite(preview_path, preview)
    print(f"Preview saved to {preview_path}  ({preview.shape[1]}×{preview.shape[0]})")

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
