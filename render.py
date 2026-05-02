#!/usr/bin/env python3
"""Render an image onto the LED matrix using calibration_result.py.

The physical display is landscape (64 wide × 40 tall). The LED map uses
standard coordinates (row=0..39 is vertical, col=0..63 is horizontal).
OpenCV loads images as BGR; NeoPixels are written directly below.
"""

import sys
import cv2
import numpy as np

try:
    from calibration_result import GRID_ORDER, BLOCK_ORIENTATION
except ImportError:
    raise SystemExit("calibration_result.py not found — run calibrate.py first")

from mapping import build_mapping
from writer import LEDS

Y_LEDS = 40   # 5 blocks × 8 rows  — vertical (height)
X_LEDS = 64   # 2 blocks × 32 cols — horizontal (width)

EDGE_CONTRAST_STRENGTH = 0.55
EDGE_DARK_DETAIL_STRENGTH = 0.85
EDGE_BLUR_SIGMA = 2.0
EDGE_DILATE_KERNEL = 5


def resize_for_leds(img: np.ndarray) -> np.ndarray:
    """Resize to LED resolution while preserving source-image edge detail."""
    frame = cv2.resize(img, (X_LEDS, Y_LEDS), interpolation=cv2.INTER_AREA)

    luma = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32)
    smooth_luma = cv2.GaussianBlur(luma, (0, 0), sigmaX=EDGE_BLUR_SIGMA)

    grad_x = cv2.Sobel(luma, cv2.CV_32F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(luma, cv2.CV_32F, 0, 1, ksize=3)
    edge = cv2.magnitude(grad_x, grad_y)
    edge = cv2.normalize(edge, None, 0.0, 1.0, cv2.NORM_MINMAX)

    kernel = np.ones((EDGE_DILATE_KERNEL, EDGE_DILATE_KERNEL), dtype=np.uint8)
    edge = cv2.dilate(edge, kernel)

    dark_detail = np.maximum(smooth_luma - luma, 0.0) * edge
    dark_detail = cv2.resize(
        dark_detail,
        (X_LEDS, Y_LEDS),
        interpolation=cv2.INTER_AREA,
    )
    edge_mask = cv2.resize(
        edge,
        (X_LEDS, Y_LEDS),
        interpolation=cv2.INTER_AREA,
    )
    edge_mask = np.clip(edge_mask, 0.0, 1.0)

    frame_f = frame.astype(np.float32)
    local_mean = cv2.GaussianBlur(frame_f, (0, 0), sigmaX=0.8)
    frame_f += (frame_f - local_mean) * edge_mask[:, :, None] * EDGE_CONTRAST_STRENGTH

    frame_luma = cv2.cvtColor(np.clip(frame_f, 0, 255).astype(np.uint8), cv2.COLOR_BGR2GRAY)
    frame_luma = frame_luma.astype(np.float32)
    target_luma = np.maximum(
        0.0,
        frame_luma - dark_detail * EDGE_DARK_DETAIL_STRENGTH,
    )
    scale = target_luma / np.maximum(frame_luma, 1.0)
    frame_f *= scale[:, :, None]
    return np.clip(frame_f, 0, 255).astype(np.uint8)


def render(image_path: str, preview_path: str = "preview.png"):
    led_map = build_mapping(GRID_ORDER, BLOCK_ORIENTATION)

    img = cv2.imread(image_path)   # BGR
    if img is None:
        raise FileNotFoundError(f"Could not load image: {image_path}")

    # Resize to landscape (64 wide × 40 tall) — cv2.resize takes (width, height)
    frame = resize_for_leds(img)

    cv2.imwrite(preview_path, frame)
    print(f"Preview saved to {preview_path}  ({frame.shape[1]}×{frame.shape[0]})")

    # Write to LEDs — convert BGR (OpenCV) → RGB for this render path.
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
