# Best known program for the n=26 circle packing problem, as found by OpenEvolve.
#
# Source: issue #156 (https://github.com/algorithmicsuperintelligence/openevolve/issues/156),
# reported by @yyh-sjtu. Discovered at iteration 206 (generation 13) of a run configured as
# phase_1 x100 -> phase_2 x100 -> phase_2 x250 with the evaluator timeout raised to 200s.
#
# The program is deterministic (no random initialization), so its result can be independently
# re-verified with the repository evaluator:
#
#   python -c "import importlib.util; spec = importlib.util.spec_from_file_location('evaluator', 'examples/circle_packing/evaluator.py'); ev = importlib.util.module_from_spec(spec); spec.loader.exec_module(ev); print(ev.evaluate('examples/circle_packing/best_program.py'))"
#
# Verified result (see best_program_info.json):
#   validity=1.0, sum_radii=2.635977394746595, target_ratio=1.0003709277975694

# EVOLVE-BLOCK-START
"""Advanced circle packing for n=26 circles using specialized patterns and optimization techniques. This version incorporates a more robust penalty function, adaptive radius adjustments during optimization, and a refined initial pattern selection strategy."""

import logging

import numpy as np
from scipy.optimize import minimize

# Configure logging (optional, but helpful for debugging)
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


def construct_packing():
    """
    Construct an optimized arrangement of 26 circles in a unit square
    that maximizes the sum of their radii using specialized patterns and optimization.

    Returns:
        Tuple of (centers, radii, sum_of_radii)
        centers: np.array of shape (26, 2) with (x, y) coordinates
        radii: np.array of shape (26) with radius of each circle
        sum_of_radii: Sum of all radii
    """
    n = 26
    best_centers = None
    best_radii = None
    best_sum = 0.0

    # Prioritize patterns known to perform well
    patterns = [
        initialize_pattern_specialized_26,
        initialize_pattern_hybrid_26,
        initialize_pattern_corner_optimized_26,
        initialize_pattern_ring_26,
        initialize_pattern_billiard_26,
        initialize_pattern_optimized_grid_26,
        initialize_pattern_greedy_26,
    ]

    for pattern_func in patterns:
        # Initialize with a specialized pattern
        centers, radii = pattern_func()

        # Ensure we have exactly 26 circles
        assert (
            centers.shape[0] == n
        ), f"Pattern function {pattern_func.__name__} returned {centers.shape[0]} circles instead of {n}"

        # Perform optimization
        try:
            centers, radii = optimize_with_scipy(centers, radii)

            # Keep the best result
            sum_radii = np.sum(radii)
            if sum_radii > best_sum:
                best_sum = sum_radii
                best_centers = centers.copy()
                best_radii = radii.copy()
        except Exception as e:
            print(f"Optimization failed for {pattern_func.__name__}: {e}")
            continue

    return best_centers, best_radii, best_sum


def initialize_pattern_specialized_26():
    """
    Initialize with a pattern specifically optimized for n=26 based on mathematical research.
    This pattern uses a hybrid approach with variable-sized circles.
    """
    n = 26
    centers = np.zeros((n, 2))
    radii = np.zeros(n)

    # Place 4 large circles in corners
    corner_radius = 0.118
    centers[0] = [corner_radius, corner_radius]
    centers[1] = [1 - corner_radius, corner_radius]
    centers[2] = [corner_radius, 1 - corner_radius]
    centers[3] = [1 - corner_radius, 1 - corner_radius]
    radii[0:4] = corner_radius

    # Place 4 medium circles at midpoints of edges
    edge_radius = 0.103
    centers[4] = [0.5, edge_radius]
    centers[5] = [0.5, 1 - edge_radius]
    centers[6] = [edge_radius, 0.5]
    centers[7] = [1 - edge_radius, 0.5]
    radii[4:8] = edge_radius

    # Place a large circle in the center
    centers[8] = [0.5, 0.5]
    radii[8] = 0.124

    # Place 8 circles in inner ring around center
    inner_radius = 0.098
    for i in range(8):
        angle = 2 * np.pi * i / 8
        dist = radii[8] + inner_radius
        centers[9 + i] = [0.5 + dist * np.cos(angle), 0.5 + dist * np.sin(angle)]
        radii[9 + i] = inner_radius

    # Place 2 circles in outer ring (adjusted from 10 to 9 to ensure we have exactly 26 circles)
    outer_radius = 0.083
    for i in range(2):
        angle = 2 * np.pi * i / 2 + np.pi / 2
        dist = radii[8] + 2 * inner_radius + outer_radius + 0.01
        centers[17 + i] = [0.5 + dist * np.cos(angle), 0.5 + dist * np.sin(angle)]
        radii[17 + i] = outer_radius

    # Place 6 circles in outer ring (adjusted from 10 to 9 to ensure we have exactly 26 circles)
    outer_radius = 0.071
    for i in range(6):
        angle = 2 * np.pi * i / 6 + np.pi / 6
        dist = 0.25
        centers[19 + i] = [0.15 + dist * np.cos(angle), 0.15 + dist * np.sin(angle)]
        radii[19 + i] = outer_radius

    # Place 3 circles in outer ring (adjusted from 10 to 9 to ensure we have exactly 26 circles)
    outer_radius = 0.076
    for i in range(3):
        angle = 2 * np.pi * i / 3 + np.pi / 3
        dist = 0.25
        centers[25 - 2 + i] = [0.85 + dist * np.cos(angle), 0.85 + dist * np.sin(angle)]
        radii[25 - 2 + i] = outer_radius

    return centers, radii


def initialize_pattern_hybrid_26():
    """
    Initialize with a hybrid pattern optimized for n=26 with strategic placement
    of variable-sized circles.
    """
    n = 26
    centers = np.zeros((n, 2))
    radii = np.zeros(n)

    # Center large circle
    centers[0] = [0.5, 0.5]
    radii[0] = 0.128

    # Inner ring (6 circles)
    inner_radius = 0.103
    for i in range(6):
        angle = 2 * np.pi * i / 6
        dist = radii[0] + inner_radius
        centers[i + 1] = [0.5 + dist * np.cos(angle), 0.5 + dist * np.sin(angle)]
        radii[i + 1] = inner_radius

    # Middle ring (8 circles, slightly offset)
    middle_radius = 0.093
    for i in range(8):
        angle = 2 * np.pi * i / 8 + np.pi / 8
        dist = radii[0] + 2 * inner_radius + middle_radius
        centers[i + 7] = [0.5 + dist * np.cos(angle), 0.5 + dist * np.sin(angle)]
        radii[i + 7] = middle_radius

    # Corner circles (4 larger circles)
    corner_radius = 0.113
    centers[15] = [corner_radius, corner_radius]
    centers[16] = [1 - corner_radius, corner_radius]
    centers[17] = [corner_radius, 1 - corner_radius]
    centers[18] = [1 - corner_radius, 1 - corner_radius]
    radii[15:19] = corner_radius

    # Edge circles (4 medium circles)
    edge_radius = 0.088
    centers[19] = [0.5, edge_radius]
    centers[20] = [0.5, 1 - edge_radius]
    centers[21] = [edge_radius, 0.5]
    centers[22] = [1 - edge_radius, 0.5]
    radii[19:23] = edge_radius

    # Fill remaining spaces with 3 smaller circles
    small_radius = 0.073
    centers[23] = [0.25, 0.25]
    centers[24] = [0.75, 0.25]
    centers[25] = [0.25, 0.75]
    radii[23:26] = small_radius

    return centers, radii


def initialize_pattern_billiard_26():
    """
    Initialize with a billiard-table inspired pattern with circles along the edges
    and a triangular pattern in the center.
    """
    n = 26
    centers = np.zeros((n, 2))
    radii = np.zeros(n)

    # 4 corner circles
    corner_radius = 0.113
    centers[0] = [corner_radius, corner_radius]
    centers[1] = [1 - corner_radius, corner_radius]
    centers[2] = [corner_radius, 1 - corner_radius]
    centers[3] = [1 - corner_radius, 1 - corner_radius]
    radii[0:4] = corner_radius

    # 8 edge circles (2 on each edge)
    edge_radius = 0.093
    edge_positions = [0.3, 0.7]

    # Bottom edge
    centers[4] = [edge_positions[0], edge_radius]
    centers[5] = [edge_positions[1], edge_radius]

    # Top edge
    centers[6] = [edge_positions[0], 1 - edge_radius]
    centers[7] = [edge_positions[1], 1 - edge_radius]

    # Left edge
    centers[8] = [edge_radius, edge_positions[0]]
    centers[9] = [edge_radius, edge_positions[1]]

    # Right edge
    centers[10] = [1 - edge_radius, edge_positions[0]]
    centers[11] = [1 - edge_radius, edge_positions[1]]

    radii[4:12] = edge_radius

    # Triangular pattern in center (14 circles)
    inner_radius = 0.083

    # Center circle
    centers[12] = [0.5, 0.5]
    radii[12] = 0.103  # Slightly larger center circle

    # Inner hexagon (6 circles)
    for i in range(6):
        angle = 2 * np.pi * i / 6
        dist = radii[12] + inner_radius
        centers[13 + i] = [0.5 + dist * np.cos(angle), 0.5 + dist * np.sin(angle)]
        radii[13 + i] = inner_radius

    # Outer partial ring (7 circles)
    outer_radius = 0.078
    for i in range(7):
        angle = 2 * np.pi * i / 7 + np.pi / 7
        dist = radii[12] + 2 * inner_radius + outer_radius
        centers[19 + i] = [0.5 + dist * np.cos(angle), 0.5 + dist * np.sin(angle)]
        radii[19 + i] = outer_radius

    return centers, radii


def initialize_pattern_corner_optimized_26():
    """
    Initialize with a pattern that emphasizes optimal corner and edge utilization.
    """
    n = 26
    centers = np.zeros((n, 2))
    radii = np.zeros(n)

    # 4 large corner circles
    corner_radius = 0.123
    centers[0] = [corner_radius, corner_radius]
    centers[1] = [1 - corner_radius, corner_radius]
    centers[2] = [corner_radius, 1 - corner_radius]
    centers[3] = [1 - corner_radius, 1 - corner_radius]
    radii[0:4] = corner_radius

    # 4 medium edge circles
    edge_radius = 0.098
    centers[4] = [0.5, edge_radius]
    centers[5] = [0.5, 1 - edge_radius]
    centers[6] = [edge_radius, 0.5]
    centers[7] = [1 - edge_radius, 0.5]
    radii[4:8] = edge_radius

    # 8 smaller edge circles
    small_edge_radius = 0.083
    centers[8] = [0.25, small_edge_radius]
    centers[9] = [0.75, small_edge_radius]
    centers[10] = [0.25, 1 - small_edge_radius]
    centers[11] = [0.75, 1 - small_edge_radius]
    centers[12] = [small_edge_radius, 0.25]
    centers[13] = [small_edge_radius, 0.75]
    centers[14] = [1 - small_edge_radius, 0.25]
    centers[15] = [1 - small_edge_radius, 0.75]
    radii[8:16] = small_edge_radius

    # Inner grid (10 circles)
    inner_radius = 0.078
    grid_positions = [0.3, 0.5, 0.7]

    count = 16
    grid_points = []
    for x in grid_positions:
        for y in grid_positions:
            grid_points.append((x, y))

    # Ensure we don't exceed 26 circles
    for x, y in grid_points[: n - count]:
        if x == 0.5 and y == 0.5:
            # Larger circle in the center
            centers[count] = [x, y]
            radii[count] = 0.113
        else:
            centers[count] = [x, y]
            radii[count] = inner_radius
        count += 1

    # Fill any remaining slots
    while count < n:
        centers[count] = [0.4 + 0.2 * (count - 16) / 10, 0.4 + 0.2 * ((count - 16) % 3) / 3]
        radii[count] = 0.073
        count += 1

    return centers, radii


def initialize_pattern_ring_26():
    """
    Initialize with concentric rings of circles, optimized for n=26.
    """
    n = 26
    centers = np.zeros((n, 2))
    radii = np.zeros(n)

    # Center circle
    centers[0] = [0.5, 0.5]
    radii[0] = 0.133

    # First ring (8 circles)
    ring1_radius = 0.098
    for i in range(8):
        angle = 2 * np.pi * i / 8
        dist = radii[0] + ring1_radius
        centers[i + 1] = [0.5 + dist * np.cos(angle), 0.5 + dist * np.sin(angle)]
        radii[i + 1] = ring1_radius

    # Second ring (12 circles)
    ring2_radius = 0.088
    for i in range(12):
        angle = 2 * np.pi * i / 12 + np.pi / 12
        dist = radii[0] + 2 * ring1_radius + ring2_radius
        centers[i + 9] = [0.5 + dist * np.cos(angle), 0.5 + dist * np.sin(angle)]
        radii[i + 9] = ring2_radius

    # Corner circles (4 circles)
    corner_radius = 0.093
    centers[21] = [corner_radius, corner_radius]
    centers[22] = [1 - corner_radius, corner_radius]
    centers[23] = [corner_radius, 1 - corner_radius]
    centers[24] = [1 - corner_radius, 1 - corner_radius]
    radii[21:25] = corner_radius

    # One extra circle
    centers[25] = [0.5, 0.15]
    radii[25] = 0.083

    return centers, radii


def initialize_pattern_optimized_grid_26():
    """
    Initializes circles in an optimized grid-like pattern, focusing on even distribution and space filling.
    """
    n = 26
    centers = np.zeros((n, 2))
    radii = np.zeros(n)

    # Start with a rough grid
    num_x = 6
    num_y = 5  # Adjusted for 26 circles

    x_coords = np.linspace(0.1, 0.9, num_x)
    y_coords = np.linspace(0.1, 0.9, num_y)

    count = 0
    for i in range(num_y):
        for j in range(num_x):
            if count < n:
                centers[count, 0] = x_coords[j]
                centers[count, 1] = y_coords[i]
                radii[count] = 0.083  # Initial radius
                count += 1
            else:
                break
        if count >= n:
            break

    # Adjust radii based on location - corners slightly larger
    for i in range(n):
        if (
            (centers[i, 0] < 0.2 and centers[i, 1] < 0.2)
            or (centers[i, 0] > 0.8 and centers[i, 1] < 0.2)
            or (centers[i, 0] < 0.2 and centers[i, 1] > 0.8)
            or (centers[i, 0] > 0.8 and centers[i, 1] > 0.8)
        ):
            radii[i] = 0.088  # Slightly larger in the corners
        elif centers[i, 0] == 0.5 and centers[i, 1] == 0.5:
            radii[i] = 0.103  # Largest radius in the very center
        else:
            radii[i] = 0.078

    return centers, radii


def initialize_pattern_greedy_26():
    """
    Initialize with a greedy approach that places larger circles first
    and then fills in with smaller ones.
    """
    n = 26
    centers = np.zeros((n, 2))
    radii = np.zeros(n)

    # Define initial positions for the first 5 circles (corners and center)
    initial_positions = [
        (0.123, 0.123),  # Bottom left corner
        (0.877, 0.123),  # Bottom right corner
        (0.123, 0.877),  # Top left corner
        (0.877, 0.877),  # Top right corner
        (0.5, 0.5),  # Center
    ]

    # Initial radii for the first 5 circles
    initial_radii = [0.123, 0.123, 0.123, 0.123, 0.143]

    for i in range(min(5, n)):
        centers[i] = initial_positions[i]
        radii[i] = initial_radii[i]

    # Place circles along the edges
    edge_positions = [
        (0.5, 0.1),  # Bottom edge
        (0.5, 0.9),  # Top edge
        (0.1, 0.5),  # Left edge
        (0.9, 0.5),  # Right edge
        (0.3, 0.1),  # Bottom edge
        (0.7, 0.1),  # Bottom edge
        (0.3, 0.9),  # Top edge
        (0.7, 0.9),  # Top edge
        (0.1, 0.3),  # Left edge
        (0.1, 0.7),  # Left edge
        (0.9, 0.3),  # Right edge
        (0.9, 0.7),  # Right edge
    ]

    edge_radii = [
        0.103,
        0.103,
        0.103,
        0.103,
        0.093,
        0.093,
        0.093,
        0.093,
        0.093,
        0.093,
        0.093,
        0.093,
    ]

    for i in range(min(len(edge_positions), n - 5)):
        centers[i + 5] = edge_positions[i]
        radii[i + 5] = edge_radii[i]

    # Place inner circles in a grid-like pattern
    inner_positions = [
        (0.3, 0.3),
        (0.5, 0.3),
        (0.7, 0.3),
        (0.3, 0.5),
        (0.7, 0.5),
        (0.3, 0.7),
        (0.5, 0.7),
        (0.7, 0.7),
        (0.4, 0.4),
    ]

    inner_radii = [0.093, 0.093, 0.093, 0.093, 0.093, 0.093, 0.093, 0.093, 0.083]

    remaining_circles = n - (5 + len(edge_positions))
    for i in range(min(len(inner_positions), remaining_circles)):
        centers[i + 5 + len(edge_positions)] = inner_positions[i]
        radii[i + 5 + len(edge_positions)] = inner_radii[i]

    return centers, radii


def optimize_with_scipy(centers, radii):
    """
    Optimize circle positions and radii using scipy.optimize.minimize.
    Uses a multi-stage approach to avoid local minima.
    """
    n = len(centers)

    # Stage 1: Optimize positions with fixed radii - Reduced iterations for speed
    def objective_positions(x):
        """Objective function for position optimization."""
        current_centers = x.reshape((n, 2))
        return calculate_penalty(current_centers, radii)

    x0_positions = centers.flatten()
    bounds_positions = [(0, 1) for _ in range(2 * n)]

    res_positions = minimize(
        objective_positions,
        x0_positions,
        method="L-BFGS-B",
        bounds=bounds_positions,
        options={"maxiter": 150, "ftol": 1e-6},  # Adjusted maxiter and ftol
    )

    improved_centers = res_positions.x.reshape((n, 2))

    # Stage 2: Optimize radii with fixed positions - Increased iterations and adjusted bounds
    def objective_radii(r):
        """Objective function for radii optimization (negative sum of radii)."""
        return -np.sum(r)

    def constraint_radii(r):
        """Constraints for radii optimization."""
        constraints = []

        # No overlapping circles
        for i in range(n):
            for j in range(i + 1, n):
                dist = np.linalg.norm(improved_centers[i] - improved_centers[j])
                constraints.append(dist - r[i] - r[j])

        # Circles within the unit square
        for i in range(n):
            constraints.append(improved_centers[i, 0] - r[i])  # x >= r
            constraints.append(1 - improved_centers[i, 0] - r[i])  # x <= 1 - r
            constraints.append(improved_centers[i, 1] - r[i])  # y >= r
            constraints.append(1 - improved_centers[i, 1] - r[i])  # y <= 1 - r

        return np.array(constraints)

    cons_radii = {"type": "ineq", "fun": constraint_radii}

    res_radii = minimize(
        objective_radii,
        radii,
        method="SLSQP",
        constraints=cons_radii,
        bounds=[(0.03, 0.2) for _ in range(n)],  # Adjusted bounds
        options={"maxiter": 500, "ftol": 1e-7},  # Increased maxiter and ftol
    )

    improved_radii = res_radii.x

    # Stage 3: Final joint optimization - Increased iterations, tighter tolerance, and refined bounds
    def objective_joint(x):
        """Objective function to minimize (negative sum of radii)."""
        current_centers = x[: 2 * n].reshape((n, 2))
        current_radii = x[2 * n :]
        return -np.sum(current_radii)

    def constraint_joint(x):
        """Constraints: no overlapping circles, circles within the unit square."""
        current_centers = x[: 2 * n].reshape((n, 2))
        current_radii = x[2 * n :]
        constraints = []

        # No overlapping circles
        for i in range(n):
            for j in range(i + 1, n):
                dist = np.linalg.norm(current_centers[i] - current_centers[j])
                constraints.append(dist - current_radii[i] - current_radii[j])

        # Circles within the unit square
        for i in range(n):
            constraints.append(current_centers[i, 0] - current_radii[i])  # x >= r
            constraints.append(1 - current_centers[i, 0] - current_radii[i])  # x <= 1 - r
            constraints.append(current_centers[i, 1] - current_radii[i])  # y >= r
            constraints.append(1 - current_centers[i, 1] - current_radii[i])  # y <= 1 - r

        return np.array(constraints)

    x0_joint = np.concatenate([improved_centers.flatten(), improved_radii])
    bounds_joint = [(0.02, 0.98) for _ in range(2 * n)] + [
        (0.03, 0.2) for _ in range(n)
    ]  # Adjusted bounds
    cons_joint = {"type": "ineq", "fun": constraint_joint}

    res_joint = minimize(
        objective_joint,
        x0_joint,
        method="SLSQP",
        constraints=cons_joint,
        bounds=bounds_joint,
        options={"maxiter": 800, "ftol": 1e-8},  # Increased maxiter and tighter tolerance
    )

    final_centers = res_joint.x[: 2 * n].reshape((n, 2))
    final_radii = res_joint.x[2 * n :]

    return final_centers, final_radii


def calculate_penalty(centers, radii):
    """
    Calculate penalty for overlapping circles or circles outside the unit square.
    Used for optimization.  Increased penalty magnitude and added radius-dependent term.
    """
    n = len(centers)
    penalty = 0.0
    overlap_penalty_factor = 300  # Increased penalty
    boundary_penalty_factor = 300  # Increased penalty

    # Penalty for overlapping circles
    for i in range(n):
        for j in range(i + 1, n):
            dist = np.linalg.norm(centers[i] - centers[j])
            overlap = radii[i] + radii[j] - dist
            if overlap > 0:
                penalty += (
                    overlap_penalty_factor * overlap**2 * (radii[i] + radii[j])
                )  # Radius-dependent penalty

    # Penalty for circles outside the unit square
    for i in range(n):
        # Left boundary
        if centers[i, 0] - radii[i] < 0:
            penalty += boundary_penalty_factor * (radii[i] - centers[i, 0]) ** 2

        # Right boundary
        if centers[i, 0] + radii[i] > 1:
            penalty += boundary_penalty_factor * (centers[i, 0] + radii[i] - 1) ** 2

        # Bottom boundary
        if centers[i, 1] - radii[i] < 0:
            penalty += boundary_penalty_factor * (radii[i] - centers[i, 1]) ** 2

        # Top boundary
        if centers[i, 1] + radii[i] > 1:
            penalty += boundary_penalty_factor * (centers[i, 1] + radii[i] - 1) ** 2

    return penalty


# This part remains fixed (not evolved)
def run_packing():
    """Run the circle packing constructor for n=26"""
    centers, radii, sum_radii = construct_packing()
    return centers, radii, sum_radii


def visualize(centers, radii):
    """
    Visualize the circle packing

    Args:
        centers: np.array of shape (n, 2) with (x, y) coordinates
        radii: np.array of shape (n) with radius of each circle
    """
    import matplotlib.pyplot as plt
    from matplotlib.patches import Circle

    fig, ax = plt.subplots(figsize=(8, 8))

    # Draw unit square
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect("equal")
    ax.grid(True)

    # Draw circles
    for i, (center, radius) in enumerate(zip(centers, radii)):
        circle = Circle(center, radius, alpha=0.5)
        ax.add_patch(circle)
        ax.text(center[0], center[1], str(i), ha="center", va="center")

    plt.title(f"Circle Packing (n={len(centers)}, sum={sum(radii):.6f})")
    # plt.show()
    plt.savefig("circle_packing_s100.png")


if __name__ == "__main__":
    centers, radii, sum_radii = run_packing()
    print(f"Sum of radii: {sum_radii}")
    # AlphaEvolve improved this to 2.635
    # Uncomment to visualize:
    visualize(centers, radii)
# EVOLVE-BLOCK-END
