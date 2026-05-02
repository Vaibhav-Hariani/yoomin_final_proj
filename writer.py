## Accepts a video file: at every frame, takes the array, writes the RGB values to the neopixel strip using the mapping in mapping.pyif PI:

X_LEDS = 64
Y_LEDS = 40

import numpy as np
from mapping import build_mapping
import cv2
try:
    import board
    import neopixel
    LEDS = neopixel.NeoPixel(
        board.D18, X_LEDS * Y_LEDS, brightness=0.1, auto_write=False)
except ImportError:
    class MOCK_LEDS:
        def __init__(self, num_leds):
            print("Using mockup LED class")
            self.leds = [(0, 0, 0)] * num_leds
            self.updated_LEDS = 0

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

    LEDS = MOCK_LEDS(X_LEDS * Y_LEDS)

def write_frame(frame: np.ndarray, LED_MAP: np.ndarray):
    if frame.shape[0] != Y_LEDS or frame.shape[1] != X_LEDS:
        print("Warning: Frame size does not match LED grid size. ")
        pass
    for i in range(Y_LEDS):
        for j in range(X_LEDS):
            led_index = LED_MAP[i][j]
            # NeoPixel (WS2812B) expects GRB order, not RGB
            r, g, b = frame[i, j]
            LEDS[led_index] = (int(g), int(r), int(b))
    LEDS.show()

def main(video_path: str, LED_MAP: np.ndarray):
    cap = cv2.VideoCapture(video_path)
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        write_frame(frame, LED_MAP)
    cap.release()

if __name__ == "__main__":
    default_order = list(range(10))
    default_orient = [0] * 10
    m = build_mapping(default_order, default_orient)
    LEDS[:] = (0, 0, 0)  # Clear all LEDs at the start
    main("output.mp4", LED_MAP=m)