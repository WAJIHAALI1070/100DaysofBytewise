#Knapsack Problem
def knapsack(values, weights, capacity):
    n = len(values)
    dp = [[0 for _ in range(capacity + 1)] for _ in range(n + 1)]

    for i in range(1, n + 1):
        for w in range(capacity + 1):
            if weights[i - 1] <= w:
                dp[i][w] = max(values[i - 1] + dp[i - 1][w - weights[i - 1]], dp[i - 1][w])
            else:
                dp[i][w] = dp[i - 1][w]

    return dp[n][capacity]


# Example usage
values = [1, 4, 5, 7]
weights = [1, 3, 4, 5]
capacity = 7
max_value = knapsack(values, weights, capacity)
print("Maximum value:", max_value)
