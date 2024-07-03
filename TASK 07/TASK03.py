#Exercise: Create a scatter plot using Matplotlib to visualize the relationship between two variables in a dataset.
import matplotlib.pyplot as plt

# Data poitns, note that X and Y must be of the same point
x = [1, 2, 3, 4, 5, 8, 19, 10,83]
y = [10, 15, 7, 10, 20, 14, 50, 90, 21]

# Create a scatter plot
plt.scatter(x, y, color='green', marker='o', label='Data Points')

# Add labels and title
plt.xlabel('X-axis label')
plt.ylabel('Y-axis label')
plt.title('Scatter Plot Example')

# Add legend
plt.legend()

# Display the plot
plt.show()
