# EVOLVE-BLOCK-START

import numpy as np

def transform_grid_attempt_1(grid):
    """
    Transformation logic:
    1. Identify the bounding box of the '8' values (mask) in the grid.
    2. The output grid has the same dimensions as this bounding box.
    3. For each cell (r, c) in the bounding box:
       a. Calculate the vertically symmetric row: target_r = (Height + 1) - r.
          If target_r is valid and grid[target_r, c] is not 8, use it.
       b. Otherwise, use the diagonally transposed position: grid[c, r].
    """
    arr = _validate_grid(grid)
    rows, cols = np.where(arr == 8)
    
    if len(rows) == 0:
        return arr
        
    min_r, max_r = np.min(rows), np.max(rows)
    min_c, max_c = np.min(cols), np.max(cols)
    
    height = max_r - min_r + 1
    width = max_c - min_c + 1
    
    out = np.zeros((height, width), dtype=np.int32)
    
    N = arr.shape[0]
    # Reflection constant determined to be N + 1 (31 for 30x30)
    reflection_constant = N + 1
    
    for r in range(min_r, max_r + 1):
        for c in range(min_c, max_c + 1):
            target_r = reflection_constant - r
            
            val = 8
            if 0 <= target_r < N:
                val = arr[target_r, c]
            
            if val == 8:
                # Fallback to diagonal transpose
                # Ensure indices are within bounds (though for square grids they should be)
                if 0 <= c < N and 0 <= r < arr.shape[1]:
                    val = arr[c, r]
            
            out[r - min_r, c - min_c] = val
            
    return out

def transform_grid_attempt_2(grid):
    """
    Similar to attempt 1 but with an additional fallback strategy.
    Strategies:
    1. Vertical reflection: grid[31-r, c]
    2. Diagonal transpose: grid[c, r]
    3. Horizontal reflection: grid[r, 31-c]
    """
    arr = _validate_grid(grid)
    rows, cols = np.where(arr == 8)
    
    if len(rows) == 0:
        return arr
        
    min_r, max_r = np.min(rows), np.max(rows)
    min_c, max_c = np.min(cols), np.max(cols)
    
    out = np.zeros((max_r - min_r + 1, max_c - min_c + 1), dtype=np.int32)
    N = arr.shape[0]
    M = arr.shape[1]
    
    for r in range(min_r, max_r + 1):
        for c in range(min_c, max_c + 1):
            val = 8
            
            # 1. Vertical reflection
            tr = (N + 1) - r
            if 0 <= tr < N:
                val = arr[tr, c]
            
            # 2. Diagonal transpose
            if val == 8:
                if 0 <= c < N and 0 <= r < M:
                    val = arr[c, r]
            
            # 3. Horizontal reflection
            if val == 8:
                tc = (M + 1) - c
                if 0 <= tc < M:
                    val = arr[r, tc]
                    
            out[r - min_r, c - min_c] = val
            
    return out

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