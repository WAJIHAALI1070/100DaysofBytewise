#Exercise: Use Matplotlib to create a subplot grid that displays multiple charts in a single figure.
import matplotlib.pyplot as plt
import numpy as np

# Generate some data
x = np.linspace(0, 10, 100)
y1 = np.sin(x)
y2 = np.cos(x)
y3 = np.tan(x)

# Create a figure and a subplot grid (3 rows, 1 column)
fig, axs = plt.subplots(3, 1, figsize=(8, 10))

# Plot data on each subplot
axs[0].plot(x, y1, color='blue')
axs[0].set_title('Sin(x)')
axs[0].set_xlabel('X')
axs[0].set_ylabel('Y')

axs[1].plot(x, y2, color='green')
axs[1].set_title('Cos(x)')
axs[1].set_xlabel('X')
axs[1].set_ylabel('Y')

axs[2].plot(x, y3, color='red')
axs[2].set_title('Tan(x)')
axs[2].set_xlabel('X')
axs[2].set_ylabel('Y')

# Adjust layout and display the plot
plt.tight_layout()
plt.show()
