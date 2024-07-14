user_string = input("Enter a string: ")
if user_string == user_string[::-1]:
    print(user_string, "is a palindrome")
else:
    print(user_string, "is not a palindrome")

