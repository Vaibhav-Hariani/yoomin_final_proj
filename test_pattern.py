#!/usr/bin/env python3
"""Render a full-display gradient for calibration verification.

Hue sweeps red→green→blue left-to-right; brightness fades top-to-bottom.
Any block in the wrong position or orientation will be obviously out of place.
"""
import numpy as np
import cv2
from calibration_result import GRID_ORDER, BLOCK_ORIENTATION
from mapping import build_mapping
from writer import LEDS

Y_LEDS = 40
X_LEDS = 64

led_map = build_mapping(GRID_ORDER, BLOCK_ORIENTATION)

hsv = np.zeros((Y_LEDS, X_LEDS, 3), dtype=np.uint8)
for y in range(Y_LEDS):
    for x in range(X_LEDS):
        hsv[y, x] = [
            int(x / X_LEDS * 179),                    # hue: left=red, right=magenta
            255,                                        # full saturation
            int(255 * (1.0 - 0.4 * y / Y_LEDS)),      # value: bright top, dimmer bottom
        ]

frame = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
cv2.imwrite("test_pattern_preview.png", frame)
print("Preview saved to test_pattern_preview.png")

for y in range(Y_LEDS):
    for x in range(X_LEDS):
        led_index = int(led_map[y, x])
        LEDS[led_index] = (int(frame[y, x, 2]),   # R
                           int(frame[y, x, 1]),   # G
                           int(frame[y, x, 0]))   # B
LEDS.show()
print("Pattern written to LEDs.")
