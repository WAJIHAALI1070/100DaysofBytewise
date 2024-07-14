def longest_palindromic_substring(s):
    n = len(s)
    if n == 0:
        return ""

    start = 0
    max_len = 1

    def expand_around_center(left, right):
        nonlocal start, max_len
        while left >= 0 and right < n and s[left] == s[right]:
            left -= 1
            right += 1
        # Calculate length of palindrome substring
        current_len = right - left - 1
        # Update start and max_len if current palindrome is longer
        if current_len > max_len:
            start = left + 1
            max_len = current_len

    for i in range(n):
        # Case 1: Palindrome centered at s[i]
        expand_around_center(i, i)
        # Case 2: Palindrome centered between s[i] and s[i+1]
        if i + 1 < n:
            expand_around_center(i, i + 1)

    return s[start:start + max_len]

# Function to input string from user and find longest palindromic substring
def find_longest_palindrome():
    input_string = input("Enter a string: ")
    if not input_string.strip():  # Check if input is empty or only whitespace
        print("Empty string entered. No palindrome found.")
        return

    longest_palindrome = longest_palindromic_substring(input_string)
    print(f"Longest palindromic substring: {longest_palindrome}")

# Example usage:
if __name__ == "__main__":
    find_longest_palindrome()
