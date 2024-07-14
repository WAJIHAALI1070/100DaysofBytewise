def merge_sorted_arrays(arr1, arr2):
    merged = []
    i = 0
    j = 0

    while i < len(arr1) and j < len(arr2):
        if arr1[i] <= arr2[j]:
            merged.append(arr1[i])
            i += 1
        else:
            merged.append(arr2[j])
            j += 1

    # Append remaining elements of arr1
    while i < len(arr1):
        merged.append(arr1[i])
        i += 1

    # Append remaining elements of arr2
    while j < len(arr2):
        merged.append(arr2[j])
        j += 1

    return merged


# Example of taking arrays at runtime
arr1 = list(map(int, input("Enter sorted array 1 (space-separated): ").split()))
arr2 = list(map(int, input("Enter sorted array 2 (space-separated): ").split()))

merged_array = merge_sorted_arrays(arr1, arr2)
print(f"Merged sorted array: {merged_array}")
