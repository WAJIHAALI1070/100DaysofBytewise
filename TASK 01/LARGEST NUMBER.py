first_number = float(input("Enter first number: "))
second_number = float(input("Enter second number: "))
third_number = float(input("Enter third number: "))

if first_number >= second_number and first_number >= third_number:
    largest_number = first_number
elif second_number >= first_number and second_number >= third_number:
    largest_number = second_number
else:
    largest_number = third_number

print("The largest number is:", largest_number)
