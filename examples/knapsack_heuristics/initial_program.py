def solve_knapsack(items, capacity):
    """
    Solves the 0/1 knapsack problem using a greedy heuristic based on value/weight ratio.

    Args:
        items: List of dicts, each with 'value' and 'weight'.
        capacity: Maximum weight capacity of the knapsack.

    Returns:
        A list of indices of the items selected for the knapsack.
    """
    # Calculate value/weight ratio for each item and store with original index
    ratios = []
    for i, item in enumerate(items):
        ratio = item["value"] / item["weight"] if item["weight"] > 0 else float("inf")
        ratios.append((ratio, i))

    # Sort items by ratio in descending order
    ratios.sort(key=lambda x: x[0], reverse=True)

    selected_indices = []
    current_weight = 0

    for ratio, index in ratios:
        item_weight = items[index]["weight"]
        if current_weight + item_weight <= capacity:
            selected_indices.append(index)
            current_weight += item_weight

    return selected_indices
