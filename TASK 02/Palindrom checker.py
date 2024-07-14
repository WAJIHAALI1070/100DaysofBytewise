import string

sentence = input("Enter a sentence: ")
# Remove spaces and punctuation, and convert to lowercase
cleaned_sentence = ''.join(char.lower() for char in sentence if char.isalnum())

if cleaned_sentence == cleaned_sentence[::-1]:
    print(f"{sentence} is a palindrome.")
else:
    print(f"{sentence} is not a palindrome.")
