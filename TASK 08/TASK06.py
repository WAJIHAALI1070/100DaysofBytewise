#Exercise: Calculate the cumulative distribution function (CDF) for the petal lengths and plot it.
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# Load the Iris dataset from Seaborn
iris = sns.load_dataset('iris')

# Extract petal length data
petal_length = iris['petal_length']

# Calculate cumulative distribution function (CDF)
sorted_data = np.sort(petal_length)
cumulative_prob = np.linspace(0, 1, len(sorted_data))

# Plotting
plt.figure(figsize=(8, 6))
plt.plot(sorted_data, cumulative_prob, marker='o', linestyle='-', color='b')
plt.title('Cumulative Distribution Function (CDF) of Petal Lengths')
plt.xlabel('Petal Length')
plt.ylabel('Cumulative Probability')
plt.grid(True)
plt.show()
