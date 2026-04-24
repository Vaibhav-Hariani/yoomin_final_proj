## Defines relationship between pixels and the LEDs
import numpy as np


## Pass in the order and orientation of each of the blocks in the grid
def build_mapping(grid_order, block_orientation):
    """Build a 32x96 mapping from pixel (row,col) to LED index.

    Grid layout:
      - The overall display is 32 rows x 96 columns.
      - It is composed of 12 blocks arranged as 4 rows x 3 cols of blocks.
      - Each block is 8 rows x 32 cols (BLOCK_HEIGHT x BLOCK_WIDTH).

    Parameters
    - grid_order: sequence of 12 integers in [0..11] specifying the block id
      (position id in row-major: id = block_row*3 + block_col) in the wiring
      order. The first element is the block that receives LED indices 0..255,
      the second receives 256..511, etc.
    - block_orientation: sequence/array of length 12 where element at index b
      is either 0 or 180. If 0: the block's local pixel order is normal
      (top-left -> left-to-right, then down). If 180: the block is rotated
      180 degrees (both axes flipped) before assigning indices.

    Returns
    - mapping: numpy.ndarray shape (32,96) with dtype=int where mapping[r,c]
      is the LED index to write for pixel (r,c).
    """
    # Constants
    BLOCK_HEIGHT = 8
    BLOCK_WIDTH = 32
    BLOCK_ROWS = 4
    BLOCK_COLS = 3

    num_blocks = BLOCK_ROWS * BLOCK_COLS
    block_size = BLOCK_HEIGHT * BLOCK_WIDTH

    # Validate inputs
    grid_order = list(grid_order)
    if len(grid_order) != num_blocks:
        raise ValueError(f"grid_order must have length {num_blocks}")
    if len(block_orientation) != num_blocks:
        raise ValueError(f"block_orientation must have length {num_blocks}")

    # Prepare output
    rows = BLOCK_ROWS * BLOCK_HEIGHT
    cols = BLOCK_COLS * BLOCK_WIDTH
    mapping = np.empty((rows, cols), dtype=int)

    # For each block in the wiring order, compute its indices and place them
    for wire_index, block_id in enumerate(grid_order):
        if not (0 <= block_id < num_blocks):
            raise ValueError("block ids in grid_order must be in 0..{num_blocks-1}")

        # block position in the full grid
        block_row = block_id // BLOCK_COLS
        block_col = block_id % BLOCK_COLS

        # local pixel indices for the block (row-major left->right, top->bottom)
        local = np.arange(block_size, dtype=int).reshape((BLOCK_HEIGHT, BLOCK_WIDTH))

        # orientation: 0 => normal, 180 => rotate 180 degrees
        orient = int(block_orientation[block_id])
        if orient not in (0, 180):
            raise ValueError("block_orientation values must be 0 or 180")
        if orient == 180:
            local = np.rot90(local, 2)

        # offset indices by how many LEDs came before this block in the chain
        start = wire_index * block_size
        local = local + start

        # place into global mapping
        r0 = block_row * BLOCK_HEIGHT
        c0 = block_col * BLOCK_WIDTH
        mapping[r0:r0 + BLOCK_HEIGHT, c0:c0 + BLOCK_WIDTH] = local

    return mapping


if __name__ == '__main__':
    # Small self-test / example
    # Default wiring: blocks wired in row-major order, all orientations 0
    default_order = list(range(12))
    default_orient = [0] * 12
    m = build_mapping(default_order, default_orient)
    print("mapping shape:", m.shape)
    # Print a small sample: top-left 8x16
    print(m[0:8, 0:16])



