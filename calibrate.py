#!/usr/bin/env python3
"""Calibration script: generate a distinct pattern per block, save image,
and write it to LEDs via write_frame.

Usage:
  python3 calibrate.py --order 0,1,2,...,11 --orient 0,0,180,...
"""
import argparse
import numpy as np
import cv2
from mapping import build_mapping
from writer import write_frame


BLOCK_HEIGHT = 8
BLOCK_WIDTH = 32
BLOCK_ROWS = 4
BLOCK_COLS = 3
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


def build_calibration_image(order, orient):
    frame = np.zeros((Y_LEDS, X_LEDS, 3), dtype=np.uint8)
    # For each block id (position), draw its pattern so we can visually confirm
    for block_id in range(BLOCK_ROWS * BLOCK_COLS):
        br = block_id // BLOCK_COLS
        bc = block_id % BLOCK_COLS
        r0 = br * BLOCK_HEIGHT
        c0 = bc * BLOCK_WIDTH
        pattern = make_block_pattern(block_id)
        frame[r0:r0 + BLOCK_HEIGHT, c0:c0 + BLOCK_WIDTH] = pattern

    return frame


def parse_list_arg(s, length, name):
    if s is None:
        return None
    parts = [p.strip() for p in s.split(',') if p.strip() != '']
    vals = [int(x) for x in parts]
    if len(vals) != length:
        raise argparse.ArgumentTypeError(f"{name} must have {length} comma-separated integers")
    return vals


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--order', help='Comma-separated wiring order (12 ints)')
    parser.add_argument('--orient', help='Comma-separated orientations (12 ints 0 or 180)')
    parser.add_argument('--out', help='Output image filename', default='calibration.png')
    args = parser.parse_args()

    default_order = list(range(BLOCK_ROWS * BLOCK_COLS))
    default_orient = [0] * (BLOCK_ROWS * BLOCK_HEIGHT // BLOCK_HEIGHT * BLOCK_COLS)

    order = parse_list_arg(args.order, BLOCK_ROWS * BLOCK_COLS, 'order') or default_order
    orient = parse_list_arg(args.orient, BLOCK_ROWS * BLOCK_COLS, 'orient') or [0] * (BLOCK_ROWS * BLOCK_COLS)

    # Build the image that labels/marks each block position
    img = build_calibration_image(order, orient)

    # Save a visualization (convert RGB -> BGR for OpenCV)
    cv2.imwrite(args.out, img[:, :, ::-1])
    print(f"Saved calibration image to {args.out}")

    # Build mapping using provided wiring order and orientations
    mapping = build_mapping(order, orient)

    # Use writer.write_frame to output to LEDs (or mock)
    print("Writing calibration frame to LEDs...")
    write_frame(img, mapping)


if __name__ == '__main__':
    main()
