#!/usr/bin/env python3
"""Render video frames onto the LED matrix using calibration_result.py.

The physical display is landscape (64 wide × 40 tall). The LED map uses
standard coordinates (row=0..39 is vertical, col=0..63 is horizontal).
OpenCV decodes video frames as BGR; NeoPixels are written directly below.
"""

import argparse
import sys
import time
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


def render(frame: np.ndarray, mapping: np.ndarray):
    """Render one BGR frame through the provided pixel-to-LED mapping."""
    if frame is None:
        raise ValueError("frame must not be None")
    if mapping.shape != (Y_LEDS, X_LEDS):
        raise ValueError(f"mapping must have shape {(Y_LEDS, X_LEDS)}, got {mapping.shape}")

    frame = cv2.resize(frame, (X_LEDS, Y_LEDS))
    frame = frame // 10

    # Write to LEDs: convert BGR (OpenCV) -> RGB for this render path.
    for y in range(Y_LEDS):
        for x in range(X_LEDS):
            led_index = int(mapping[y, x])
            # R = int(frame[y, x, 2])
            # G = int(frame[y, x, 1])
            # B = int(frame[y, x, 0])
            # if R < 50 and G < 50 and B < 50:
            #     print(f"Dark pixel found at ({x}, {y}): R={R}, G={G}, B={B}")
            LEDS[led_index] = (int(frame[y, x, 2]),   # R
                               int(frame[y, x, 1]),   # G
                               int(frame[y, x, 0]))   # B
    LEDS.show()


def render_video(video_path: str, mapping: np.ndarray, *, max_frames: int | None = None):
    """Decode a video and render it to the LEDs frame by frame."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_delay = 1.0 / fps if fps and fps > 0 else 0.0
    frame_count = 0

    try:
        while True:
            if max_frames is not None and frame_count >= max_frames:
                break

            ok, frame = cap.read()
            if not ok:
                break

            started = time.monotonic()
            render(frame, mapping)
            frame_count += 1

            if frame_delay:
                elapsed = time.monotonic() - started
                if elapsed < frame_delay:
                    time.sleep(frame_delay - elapsed)
    finally:
        cap.release()

    print(f"Rendered {frame_count} frame(s) from {video_path}.")


def main(argv: list[str] | None = None):
    parser = argparse.ArgumentParser(description="Render a video onto the LED matrix.")
    parser.add_argument("video_path", nargs="?", default="output.mp4")
    parser.add_argument("--max-frames", type=int, default=None)
    args = parser.parse_args(argv)

    mapping = build_mapping(GRID_ORDER, BLOCK_ORIENTATION)
    render_video(args.video_path, mapping, max_frames=args.max_frames)


if __name__ == "__main__":
    main(sys.argv[1:])
