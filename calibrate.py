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


def build_calibration_image():
    """Build a 5x2 block calibration image with distinct colors per block.
    
    Each physical block position gets a unique color based on its row-major
    block ID (0-9), independent of wiring order.
    """
    frame = np.zeros((Y_LEDS, X_LEDS, 3), dtype=np.uint8)
    
    for block_id in range(BLOCK_ROWS * BLOCK_COLS):
        block_row = block_id // BLOCK_COLS
        block_col = block_id % BLOCK_COLS
        r0 = block_row * BLOCK_HEIGHT
        c0 = block_col * BLOCK_WIDTH
        pattern = make_block_pattern(block_id)
        frame[r0:r0 + BLOCK_HEIGHT, c0:c0 + BLOCK_WIDTH] = pattern
    
    return frame


def get_wiring_order():
    """Return the wiring order for the 5x2 block matrix.
    
    LED chain physically goes: Column 0 top→bottom, then Column 1 bottom→top.
    In row-major block IDs: [0, 2, 4, 6, 8, 9, 7, 5, 3, 1]
    This maps physical block position to its position in the LED chain.
    """
    return [0, 2, 4, 6, 8, 9, 7, 5, 3, 1]



def main():
    """Run calibration with hard-coded 5x2 block structure."""
    # Build the calibration image with simple row-major layout
    img = build_calibration_image()

    # Save a visualization (convert RGB -> BGR for OpenCV)
    out_filename = 'calibration.png'
    cv2.imwrite(out_filename, img[:, :, ::-1])
    print(f"Saved calibration image to {out_filename}")

    # Build mapping using hard-coded wiring order and orientations
    order = get_wiring_order()
    orient = [0] * (BLOCK_ROWS * BLOCK_COLS)
    mapping = build_mapping(order, orient)

    # Use writer.write_frame to output to LEDs (or mock)
    print("Writing calibration frame to LEDs...")
    write_frame(img, mapping)


if __name__ == '__main__':
    main()
