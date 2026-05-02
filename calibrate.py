#!/usr/bin/env python3
"""Calibration script: generate a distinct pattern per block, save image,
and write it to LEDs via write_frame.

Hard-coded 5x2 LED block matrix with wiring:
  Column 0: Blocks 0-4 (top to bottom)
  Column 1: Blocks 5-9 (bottom to top, i.e., reverse wiring)
"""
import numpy as np
import cv2
from mapping import build_mapping
from writer import write_frame


# Hard-coded 5x2 block LED matrix
BLOCK_HEIGHT = 8
BLOCK_WIDTH = 32
BLOCK_ROWS = 5
BLOCK_COLS = 2
Y_LEDS = BLOCK_ROWS * BLOCK_HEIGHT
X_LEDS = BLOCK_COLS * BLOCK_WIDTH


def make_block_pattern(block_id):
    """Return an (H,W,3) RGB uint8 array with a distinct, non-symmetric pattern.
    """
    H, W = BLOCK_HEIGHT, BLOCK_WIDTH
    # Base color per block (RGB)
    r = (block_id * 53) % 256
    g = (block_id * 97) % 256
    b = (block_id * 163) % 256
    block = np.zeros((H, W, 3), dtype=np.uint8)
    block[:, :, 0] = r
    block[:, :, 1] = g
    block[:, :, 2] = b

    # Add a slanted bright stripe (white-ish) whose offset depends on block_id
    for c in range(W):
        # slope across block height
        row = (c * H) // W
        row = (row + (block_id % H)) % H
        block[row, c] = np.array([255, 255, 255], dtype=np.uint8)

    # Add a small marker rectangle in a corner to break symmetry
    corner = block_id % 4
    mark_h, mark_w = max(1, H // 4), max(3, W // 8)
    vpad = (block_id // 4)  # small variation per group
    if corner == 0:  # top-left
        r0, c0 = 0 + vpad, 0 + vpad
    elif corner == 1:  # top-right
        r0, c0 = 0 + vpad, W - mark_w - vpad
    elif corner == 2:  # bottom-left
        r0, c0 = H - mark_h - vpad, 0 + vpad
    else:  # bottom-right
        r0, c0 = H - mark_h - vpad, W - mark_w - vpad
    block[r0:r0 + mark_h, c0:c0 + mark_w] = np.array([0, 0, 0], dtype=np.uint8)

    return block


def build_calibration_image(mapping):
    """Build a calibration image that matches exactly what will appear on LEDs.
    
    For each physical pixel position, look up its LED index in the mapping,
    then color it based on which block of the LED chain it belongs to.
    This ensures the saved image matches what the LEDs will display.
    """
    frame = np.zeros((Y_LEDS, X_LEDS, 3), dtype=np.uint8)
    
    # For each pixel, determine its LED index and color it based on block in LED chain
    for i in range(Y_LEDS):
        for j in range(X_LEDS):
            led_index = mapping[i, j]
            # Which block in the LED chain? (each block is 256 LEDs)
            chain_block = led_index // 256
            pattern = make_block_pattern(chain_block)
            # Use a single pixel from the pattern for this position
            frame[i, j] = pattern[i % BLOCK_HEIGHT, j % BLOCK_WIDTH]
    
    return frame


def get_wiring_order():
    """Return the wiring order for the 5x2 block matrix.
    
    Maps physical position to block index:
      Column 0: Block 0, 1, 2, 3, 4 (top to bottom)
      Column 1: Block 9, 8, 7, 6, 5 (top to bottom in layout, but reverse wiring)
    """
    order = []
    # Column 0: top to bottom (blocks 0-4)
    for row in range(BLOCK_ROWS):
        order.append(row)
    # Column 1: bottom to top (blocks 9,8,7,6,5)
    for row in range(BLOCK_ROWS):
        order.append(9 - row)
    return order



def main():
    """Run calibration with hard-coded 5x2 block structure."""
    # Hard-coded wiring order for 5x2 block matrix
    order = get_wiring_order()
    orient = [0] * (BLOCK_ROWS * BLOCK_COLS)
    
    # Build mapping first
    mapping = build_mapping(order, orient)
    
    # Build the calibration image using the mapping so it matches what LEDs will display
    img = build_calibration_image(mapping)

    # Save a visualization (convert RGB -> BGR for OpenCV)
    out_filename = 'calibration.png'
    cv2.imwrite(out_filename, img[:, :, ::-1])
    print(f"Saved calibration image to {out_filename}")

    # Use writer.write_frame to output to LEDs (or mock)
    print("Writing calibration frame to LEDs...")
    write_frame(img, mapping)


if __name__ == '__main__':
    main()
