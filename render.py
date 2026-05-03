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
import numpy as np

X_LEDS = 64
Y_LEDS = 40

import cv2
try:
    import board
    import neopixel

except ImportError:
    print("Warning: board and neopixel libraries not found. Using mockup LED class.")
    class MOCK_LEDS:
        def __init__(self, num_leds, brightness=1.0):
            print("Using mockup LED class")
            self.leds = [(0, 0, 0)] * num_leds
            self.updated_LEDS = 0
            self.brightness = brightness

        def __setitem__(self, index, color):
            if 0 <= index < len(self.leds):
                self.leds[index] = color
                self.updated_LEDS += 1

        def fill(self, color):
            self.leds = [color] * len(self.leds)

        def show(self):
            print(f" {self.updated_LEDS} LEDS Have been Updated!")
            self.updated_LEDS = 0
            # print("LED colors updated (simulation):")
        def clear(self):
            pass

Y_LEDS = 40   # 5 blocks × 8 rows  — vertical (height)
X_LEDS = 64   # 2 blocks × 32 cols — horizontal (width)

def render(frame: np.ndarray, mapping: np.ndarray, LEDS, brightness: float = 1.0):
    """Render one BGR frame through the provided pixel-to-LED mapping."""
    if frame is None:
        raise ValueError("frame must not be None")
    if mapping.shape != (Y_LEDS, X_LEDS):
        raise ValueError(f"mapping must have shape {(Y_LEDS, X_LEDS)}, got {mapping.shape}")

    # Write to LEDs: convert BGR (OpenCV) -> RGB for this render path.
    # Apply brightness scaling manually so MOCK_LEDS and real NeoPixel behave the same
    b = float(brightness)
    for y in range(Y_LEDS):
        for x in range(X_LEDS):
            led_index = int(mapping[y, x])
            r = int(np.clip(frame[y, x, 2] * b, 0, 255))
            g = int(np.clip(frame[y, x, 1] * b, 0, 255))
            bl = int(np.clip(frame[y, x, 0] * b, 0, 255))
            LEDS[led_index] = (r, g, bl)
    LEDS.show()


def render_video(LEDS, video_path: str, mapping: np.ndarray, max_frames = None, div=5, brightness: float = 1.0 ):
    """Decode a video and render it to the LEDs frame by frame."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_delay = 1.0 / (5 * fps) if fps and fps > 0 else 0.0
    frame_count = 0

    try:
        while True:
            if max_frames is not None and frame_count >= max_frames:
                break

            ok, frame = cap.read()
            if not ok:
                break

            started = time.monotonic()
            frame = cv2.resize(frame, (X_LEDS, Y_LEDS))
            frame = frame // div
            render(frame, mapping, LEDS, brightness=brightness)
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
    parser.add_argument("--brightness", type=float, default=1)
    parser.add_argument("--div", type=int, default=5)

    args = parser.parse_args(argv)
    try:
        # We will apply brightness scaling manually in `render`, so create the
        # real NeoPixel object with full brightness here to avoid double-scaling.
        LEDS = neopixel.NeoPixel(
            board.D18, X_LEDS * Y_LEDS, brightness=1.0, auto_write=False)
    except:
        LEDS = MOCK_LEDS(X_LEDS * Y_LEDS, brightness=args.brightness)

    mapping = build_mapping(GRID_ORDER, BLOCK_ORIENTATION)
    render_video(LEDS, args.video_path, mapping, max_frames=args.max_frames, div = args.div, brightness=args.brightness)


if __name__ == "__main__":
    while True:
        main(sys.argv[1:])
