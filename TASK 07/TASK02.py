#Exercise: Plot a bar chart using Matplotlib to show the frequency of different categories in a dataset.
import matplotlib.pyplot as plt

# Generating random data
categories = ['LUMS', 'FAST', 'COMSATS', 'NUST']
values = [30, 40, 20, 50]

# Create a bar chart
plt.bar(categories, values, color='skyblue')

# Add labels and title
plt.xlabel('Categories')
plt.ylabel('Successful Engineers')
plt.title('Bar Chart Example')

# Display the plot
plt.show()
