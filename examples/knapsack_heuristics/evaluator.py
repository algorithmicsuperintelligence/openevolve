import importlib.util
import os
import sys
import random


def solve_knapsack_dp(items, capacity):
    """Solves the 0/1 knapsack problem using dynamic programming to find the optimal value."""
    n = len(items)
    dp = [[0 for _ in range(capacity + 1)] for _ in range(n + 1)]

    for i in range(1, n + 1):
        for w in range(1, capacity + 1):
            if items[i - 1]["weight"] <= w:
                dp[i][w] = max(
                    items[i - 1]["value"] + dp[i - 1][w - items[i - 1]["weight"]],
                    dp[i - 1][w],
                )
            else:
                dp[i][w] = dp[i - 1][w]
    return dp[n][capacity]


def generate_difficult_instance(n, weight_range=(10, 100), correlation="strong"):
    """
    Generates a difficult knapsack instance.
    "strong": value = weight + 10 (hard for some, but greedy ratio still okay)
    "weak": value = weight + random(-5, 5)
    "uncorrelated": value and weight independent
    "inverse": large weights have slightly better value but worse ratio
    """
    items = []
    total_weight = 0
    for _ in range(n):
        w = random.randint(*weight_range)
        if correlation == "strong":
            v = w + 10
        elif correlation == "weak":
            v = max(1, w + random.randint(-5, 5))
        elif correlation == "uncorrelated":
            v = random.randint(10, 100)
        elif correlation == "inverse":
            # Higher weight, higher value, but ratio might be tricky
            v = int(w * (0.8 + random.random() * 0.4))
        else:
            v = random.randint(10, 100)
        items.append({"value": v, "weight": w})
        total_weight += w

    capacity = int(total_weight * 0.3)  # Even tighter capacity
    optimal_value = solve_knapsack_dp(items, capacity)
    return {"capacity": capacity, "items": items, "optimal_value": optimal_value}


def get_test_instances():
    """Returns a list of difficult knapsack problem instances."""
    # Seed for reproducibility
    random.seed(42)

    instances = [
        # 1. Tricky small instance (Greedy trap)
        {
            "name": "greedy_trap_small",
            "capacity": 100,
            "items": [
                {"value": 10, "weight": 60},
                {"value": 10, "weight": 60},
                {"value": 12, "weight": 100},
            ],
            "optimal_value": 12,
        },
        # 2. Uncorrelated (Medium)
        {
            **generate_difficult_instance(30, correlation="uncorrelated"),
            "name": "uncorrelated_medium",
        },
        # 3. Inverse correlated (Hard for greedy)
        {
            **generate_difficult_instance(40, correlation="inverse"),
            "name": "inverse_correlated_hard",
        },
        # 4. Strongly correlated (classic hard)
        {
            **generate_difficult_instance(20, correlation="strong"),
            "name": "strongly_correlated_hard",
        },
        # 5. Large scale uncorrelated
        {
            **generate_difficult_instance(
                200, weight_range=(5, 50), correlation="uncorrelated"
            ),
            "name": "large_scale_unclorrelated",
        },
        # 6. Another greedy trap
        {
            "name": "greedy_trap_medium",
            "capacity": 50,
            "items": [
                {"value": 31, "weight": 30},
                {"value": 20, "weight": 25},
                {"value": 20, "weight": 25},
            ],
            "optimal_value": 40,
        },
    ]
    return [i for i in instances if i["optimal_value"] > 0]


def evaluate(module_path):
    """
    Evaluates the solve_knapsack function in the given module.
    """
    # Load the module dynamically
    spec = importlib.util.spec_from_file_location("evolved_module", module_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    if not hasattr(module, "solve_knapsack"):
        return {
            "combined_score": 0.0,
            "correctness": 0.0,
            "error": "Function solve_knapsack not found",
        }

    test_instances = get_test_instances()
    total_score = 0
    correct_count = 0
    instance_scores = {}
    for instance in test_instances:
        instance_score = 0.0
        try:
            selected_indices = module.solve_knapsack(
                instance["items"], instance["capacity"]
            )

            total_value = 0
            total_weight = 0
            valid_indices = True

            if not isinstance(selected_indices, (list, tuple)):
                valid_indices = False
            elif len(set(selected_indices)) != len(selected_indices):
                valid_indices = False

            if valid_indices:
                for idx in selected_indices:
                    if not (0 <= idx < len(instance["items"])):
                        valid_indices = False
                        break
                    total_value += instance["items"][idx]["value"]
                    total_weight += instance["items"][idx]["weight"]

            if valid_indices and total_weight <= instance["capacity"]:
                correct_count += 1
                # Normalized by the actual optimal value found by DP
                instance_score = min(1.0, total_value / instance["optimal_value"])
            else:
                instance_score = 0.0

        except Exception:
            # Failed to execute
            instance_score = 0.0

        total_score += instance_score
        instance_scores[instance["name"]] = instance_score

    num_instances = len(test_instances)
    combined_score = total_score / num_instances if num_instances > 0 else 0
    correctness = correct_count / num_instances if num_instances > 0 else 0

    return {
        "combined_score": combined_score,
        "correctness": correctness,
        "instance_scores": instance_scores,
    }


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python evaluator.py <module_path>")
        sys.exit(1)

    metrics = evaluate(sys.argv[1])
    print(
        f"combined_score={metrics['combined_score']:.4f}, correctness={metrics['correctness']:.4f}"
    )
    print(f"instance_scores={metrics['instance_scores']}")
