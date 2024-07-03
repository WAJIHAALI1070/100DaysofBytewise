import matplotlib.pyplot as plt

#Creating random data
x = [1, 2, 3, 4, 5]
y = [10, 15, 7, 10, 20]

# Create a line plot
plt.plot(x, y)

# Add labels and title
plt.xlabel('X-axis label')
plt.ylabel('Y-axis label')
plt.title('Simple Line Plot')

# Display the plot
plt.show()
