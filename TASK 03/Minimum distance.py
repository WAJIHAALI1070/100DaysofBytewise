def min_edit_distance(str1, str2):
    m = len(str1)
    n = len(str2)

    # Create a 2D array to store the minimum edit distances
    dp = [[0] * (n + 1) for _ in range(m + 1)]

    # Initialize the dp array based on base cases
    for i in range(m + 1):
        dp[i][0] = i  # number of deletions (delete all from str1)
    for j in range(n + 1):
        dp[0][j] = j  # number of insertions (insert all to str1)

    # Fill the dp array
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if str1[i - 1] == str2[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]  # characters match, no operation needed
            else:
                dp[i][j] = 1 + min(dp[i - 1][j],  # deletion
                                   dp[i][j - 1],  # insertion
                                   dp[i - 1][j - 1]  # substitution
                                   )

    # The bottom-right cell of the dp table contains the minimum edit distance
    return dp[m][n]


# Main program to take input from the user
if __name__ == "__main__":
    str1 = input("Enter the first string: ").strip()
    str2 = input("Enter the second string: ").strip()

    result = min_edit_distance(str1, str2)
    print(f"The minimum number of operations required to transform '{str1}' into '{str2}' is: {result}")