#!/usr/bin/env python3
"""Calibration pipeline for the 5×2 LED block matrix.

Lights each display node (chain segment) one at a time. For each, enter the
corresponding image node (grid position) and orientation. Builds the wiring
map node by node, then saves the result.

Image node layout (row-major):
  Col 0  Col 1
  [0]    [1]    ← row 0
  [2]    [3]    ← row 1
  [4]    [5]    ← row 2
  [6]    [7]    ← row 3
  [8]    [9]    ← row 4
"""

import numpy as np
import cv2
from mapping import build_mapping
from writer import LEDS

BLOCK_HEIGHT = 8
BLOCK_WIDTH = 32
BLOCK_ROWS = 5
BLOCK_COLS = 2
BLOCK_SIZE = BLOCK_HEIGHT * BLOCK_WIDTH
NUM_BLOCKS = BLOCK_ROWS * BLOCK_COLS
Y_LEDS = BLOCK_ROWS * BLOCK_HEIGHT
X_LEDS = BLOCK_COLS * BLOCK_WIDTH


# ── Reference calibration image ─────────────────────────────────────────────

def make_block_pattern(block_id: int) -> np.ndarray:
    """Return an (H, W, 3) RGB uint8 array with a distinct, non-symmetric pattern."""
    H, W = BLOCK_HEIGHT, BLOCK_WIDTH
    block = np.zeros((H, W, 3), dtype=np.uint8)
    block[:, :] = [(block_id * 53) % 256, (block_id * 97) % 256, (block_id * 163) % 256]

    for c in range(W):
        block[((c * H) // W + block_id) % H, c] = [255, 255, 255]

    corner = block_id % 4
    mh, mw = max(1, H // 4), max(3, W // 8)
    vpad = block_id // 4
    offsets = [(vpad, vpad), (vpad, W - mw - vpad), (H - mh - vpad, vpad), (H - mh - vpad, W - mw - vpad)]
    r0, c0 = offsets[corner]
    block[r0:r0 + mh, c0:c0 + mw] = 0
    return block


def build_calibration_image() -> np.ndarray:
    """Build the 5×2 reference calibration image with a distinct pattern per block."""
    frame = np.zeros((Y_LEDS, X_LEDS, 3), dtype=np.uint8)
    for block_id in range(NUM_BLOCKS):
        r0 = (block_id // BLOCK_COLS) * BLOCK_HEIGHT
        c0 = (block_id % BLOCK_COLS) * BLOCK_WIDTH
        frame[r0:r0 + BLOCK_HEIGHT, c0:c0 + BLOCK_WIDTH] = make_block_pattern(block_id)
    return frame


# ── Direct LED control (no mapping) ─────────────────────────────────────────

def light_display_node(display_node: int):
    """Light only the LEDs in the given display node (chain position 0–9).

    Renders a brightness gradient along the LED chain: the first LED in the
    segment is brightest (white), the last is dim. This makes orientation
    visually clear — the bright corner is chain index 0 within the block
    (top-left at 0°, bottom-right at 180°).
    """
    LEDS.fill((0, 0, 0))
    start = display_node * BLOCK_SIZE
    for i in range(BLOCK_SIZE):
        # Linear ramp: 230 → 30 across the block
        v = 230 - (i * 200 // BLOCK_SIZE)
        LEDS[start + i] = (v, v, v)  # GRB — equal channels = white/grey
    LEDS.show()


def clear_leds():
    LEDS.fill((0, 0, 0))
    LEDS.show()


# ── Calibration session ──────────────────────────────────────────────────────

class CalibrationSession:
    """Accumulates (display_node → image_node, orientation) associations one by one.

    Call add_node() for each block as you identify it, then build_mapping()
    once all NUM_BLOCKS nodes are mapped.
    """

    def __init__(self):
        # _map[display_node] = (image_node, orientation)
        self._map = {}

    def add_node(self, *, display_node: int, image_node: int, orientation: int = 0):
        """Record that the given display_node corresponds to image_node at orientation."""
        if not (0 <= display_node < NUM_BLOCKS):
            raise ValueError(f"display_node must be 0–{NUM_BLOCKS - 1}, got {display_node}")
        if not (0 <= image_node < NUM_BLOCKS):
            raise ValueError(f"image_node must be 0–{NUM_BLOCKS - 1}, got {image_node}")
        if orientation not in (0, 180):
            raise ValueError(f"orientation must be 0 or 180, got {orientation}")
        if display_node in self._map:
            raise ValueError(f"display_node {display_node} already mapped")
        used = {v[0] for v in self._map.values()}
        if image_node in used:
            raise ValueError(f"image_node {image_node} already used by another display_node")
        self._map[display_node] = (image_node, orientation)

    def is_complete(self) -> bool:
        return len(self._map) == NUM_BLOCKS

    def pending_display_nodes(self) -> list:
        return [d for d in range(NUM_BLOCKS) if d not in self._map]

    def build_mapping(self) -> np.ndarray:
        """Assemble and return the full LED mapping once all nodes are entered."""
        if not self.is_complete():
            raise RuntimeError(f"Incomplete — pending display nodes: {self.pending_display_nodes()}")
        # grid_order[display_node] = image_node
        grid_order = [self._map[d][0] for d in range(NUM_BLOCKS)]
        # block_orientation indexed by image_node (as mapping.py expects)
        block_orientation = [0] * NUM_BLOCKS
        for _, (img, orient) in self._map.items():
            block_orientation[img] = orient
        return build_mapping(grid_order, block_orientation)

    def to_dict(self) -> dict:
        grid_order = [self._map[d][0] for d in range(NUM_BLOCKS)]
        block_orientation = [0] * NUM_BLOCKS
        for _, (img, orient) in self._map.items():
            block_orientation[img] = orient
        return {"grid_order": grid_order, "block_orientation": block_orientation}


# ── Interactive CLI ──────────────────────────────────────────────────────────

_GRID_DIAGRAM = "\n".join(
    "  " + "  ".join(f"[{row * BLOCK_COLS + col}]" for col in range(BLOCK_COLS))
    for row in range(BLOCK_ROWS)
)


def run_interactive(session=None) -> CalibrationSession:
    """Step through each unmapped display node, prompting for image_node + orientation.

    Pass an existing CalibrationSession to resume a partial calibration.
    """
    if session is None:
        session = CalibrationSession()

    print("=== LED Matrix Calibration ===")
    print("Image node grid (row-major):")
    print(_GRID_DIAGRAM)
    print()
    print("Each block shows a gradient: bright corner = chain start (LED 0 of that segment).")
    print("  0°  → bright corner is top-left")
    print("  180° → bright corner is bottom-right")
    print()
    print("For each lit display node, enter:  <image_node> <orientation>")
    print("  image_node  — which grid position lit up (0–9)")
    print("  orientation — 0 or 180  [default: 0]")
    print()

    for display_node in session.pending_display_nodes():
        light_display_node(display_node)
        print(f"Display node {display_node} is lit  "
              f"(LEDs {display_node * BLOCK_SIZE}–{(display_node + 1) * BLOCK_SIZE - 1})")

        while True:
            try:
                parts = input("  > ").split()
                image_node = int(parts[0])
                orientation = int(parts[1]) if len(parts) > 1 else 0
                session.add_node(
                    display_node=display_node,
                    image_node=image_node,
                    orientation=orientation,
                )
                break
            except (ValueError, IndexError, RuntimeError) as exc:
                print(f"  Error: {exc}  — try again")

        img, ori = session._map[display_node]
        print(f"  ✓ display_node {display_node} → image_node {img}, orientation {ori}\n")

    clear_leds()
    return session


# ── Persist calibration ──────────────────────────────────────────────────────

def save_calibration(session: CalibrationSession, path: str = "calibration_result.py"):
    """Write GRID_ORDER and BLOCK_ORIENTATION to an importable Python file."""
    data = session.to_dict()
    with open(path, "w") as f:
        f.write("# Auto-generated calibration result\n")
        f.write(f"GRID_ORDER = {data['grid_order']}\n")
        f.write(f"BLOCK_ORIENTATION = {data['block_orientation']}\n")
    print(f"Saved calibration to {path}")


# ── Entry point ──────────────────────────────────────────────────────────────

def main():
    ref = build_calibration_image()
    cv2.imwrite("calibration.png", ref[:, :, ::-1])
    print("Reference image saved to calibration.png\n")

    session = run_interactive()
    mapping = session.build_mapping()
    print(f"Mapping built: shape {mapping.shape}")
    save_calibration(session)
    return mapping


if __name__ == "__main__":
    main()
