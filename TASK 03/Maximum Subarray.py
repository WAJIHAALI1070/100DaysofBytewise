def max_subarray_sum(arr):
    # Initialize variables
    max_current = max_global = arr[0]

    for num in arr[1:]:
        max_current = max(num, max_current + num)
        if max_current > max_global:
            max_global = max_current

    return max_global


# Function to take input from the user
def get_user_input():
    # Take input from the user
    input_string = input("Enter the elements of the array, separated by spaces: ")
    # Convert the input string to a list of integers
    arr = list(map(int, input_string.split()))
    return arr


# Main program
if __name__ == "__main__":
    arr = get_user_input()
    result = max_subarray_sum(arr)
    print(f"The maximum sum of a contiguous subarray is: {result}")