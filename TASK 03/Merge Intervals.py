def input_intervals():
    intervals = []
    # Prompt for input and split into parts
    parts = input("Enter number of intervals followed by start and end points separated by spaces: ").split()

    # First part is the number of intervals
    n = int(parts[0])

    # Validate if the number of parts match expected
    if len(parts[1:]) % 2 != 0:
        raise ValueError("Invalid input format")

    # Iterate through remaining parts in pairs (start, end)
    for i in range(n):
        start, end = int(parts[2 * i + 1]), int(parts[2 * i + 2])
        intervals.append([start, end])
    return intervals


# Example usage:
if __name__ == "__main__":
    try:
        # Input intervals from user dynamically
        intervals = input_intervals()

        # Call merge_intervals function
        merged_intervals = merge_intervals(intervals)

        # Print the merged intervals
        print("Merged intervals:", merged_intervals)

    except ValueError as e:
        print(f"Error: {e}")
