n = int(input("Enter the term number: "))
a, b = 0, 1
for _ in range(n-1):
    a, b = b, a + b
print(f"The {n}th Fibonacci number is {a}.")
