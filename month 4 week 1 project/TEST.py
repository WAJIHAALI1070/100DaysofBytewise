import cv2
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model

# Load the saved model
model = load_model('mnist_ann_model.h5')  # Use the correct model name

# Load your image
img_path = 'W:\BHAYYAAAAA\DL/ganda sa eight lagta hai.png'
img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

# Check if the image was loaded correctly
if img is None:
    print(f"Error: Unable to load image at path: {img_path}")
    exit()

# Preprocess the image
img = cv2.resize(img, (28, 28))  # Resize to 28x28 pixels
img = cv2.bitwise_not(img)  # Invert the colors if necessary
img = img.astype('float32') / 255  # Normalize pixel values to [0, 1]
img = img.reshape(1, 28, 28, 1)  # Reshape to match model input

# Show the image
plt.imshow(img[0].reshape(28, 28), cmap='gray')
plt.title('Your Handwritten Digit')
plt.show()

# Predict the digit
prediction = model.predict(img)
predicted_digit = np.argmax(prediction)

print(f'Predicted digit: {predicted_digit}')
