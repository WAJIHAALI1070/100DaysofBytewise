#Longest Common Subsequence (LCS)
def lcs(X, Y):
    m = len(X)
    n = len(Y)
    dp = [[None] * (n + 1) for i in range(m + 1)]

    for i in range(m + 1):
        for j in range(n + 1):
            if i == 0 or j == 0:
                dp[i][j] = 0
            elif X[i - 1] == Y[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])

    index = dp[m][n]
    lcs = [""] * (index + 1)
    lcs[index] = ""

    i = m
    j = n
    while i > 0 and j > 0:
        if X[i - 1] == Y[j - 1]:
            lcs[index - 1] = X[i - 1]
            i -= 1
            j -= 1
            index -= 1
        elif dp[i - 1][j] > dp[i][j - 1]:
            i -= 1
        else:
            j -= 1

    return "".join(lcs)


# Example usage
X = "AGGTAB"
Y = "GXTXAYB"
result = lcs(X, Y)
print("Longest Common Subsequence:", result)
