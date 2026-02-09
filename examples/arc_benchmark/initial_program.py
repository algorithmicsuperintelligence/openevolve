# EVOLVE-BLOCK-START

import numpy as np

def transform_grid_attempt_1(grid):
    """
    Example transformation:
    - Validate input (2D, integer values 0-9).
    - Rotate the grid 90 degrees clockwise.
    - Increment every cell by 1 modulo 10 (keeps values 0-9).
    Returns a new numpy int array.
    """
    arr = _validate_grid(grid)
    out = np.rot90(arr, k=-1)  # 90 degrees clockwise
    out = (out + 1) % 10
    return out.astype(np.int32)

def transform_grid_attempt_2(grid):
    """
    Example transformation:
    - Validate input (2D, integer values 0-9).
    - Upsample each cell to a 2x2 block (doubling both dimensions).
    - Invert colors by mapping v -> 9 - v (keeps values 0-9).
    Returns a new numpy int array.
    """
    arr = _validate_grid(grid)
    out = np.repeat(np.repeat(arr, 2, axis=0), 2, axis=1)
    out = 9 - out
    return out.astype(np.int32)

# EVOLVE-BLOCK-END

def _validate_grid(grid):
    arr = np.asarray(grid)
    if arr.ndim != 2:
        raise ValueError("Input must be a 2D array.")
    # cast to integer type for value checks
    if not np.issubdtype(arr.dtype, np.integer):
        arr = arr.astype(int)
    if arr.size and (arr.min() < 0 or arr.max() > 9):
        raise ValueError("Array values must be integers in the range 0-9.")
    return arr