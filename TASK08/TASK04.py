#Exercise: Define a random variable for the sepal length and calculate the probability distribution of sepal lengths.
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import norm

# Load the Iris dataset from Seaborn
iris = sns.load_dataset('iris')

# Define the random variable (Sepal Length)
sepal_length = iris['sepal_length']

# Fit a normal distribution to the data
mu, std_dev = norm.fit(sepal_length)

# Generate points along the x-axis for the PDF plot
x = np.linspace(sepal_length.min(), sepal_length.max(), 100)
pdf = norm.pdf(x, mu, std_dev)

# Plotting the probability distribution
plt.figure(figsize=(8, 6))
plt.hist(sepal_length, bins=20, density=True, alpha=0.6, color='g', label='Histogram of Sepal Lengths')
plt.plot(x, pdf, 'k-', linewidth=2, label=r'Normal Fit ($\mu$={:.2f}, $\sigma$={:.2f})'.format(mu, std_dev))
plt.title('Probability Distribution of Sepal Lengths')
plt.xlabel('Sepal Length')
plt.ylabel('Probability Density')
plt.legend()
plt.grid(True)
plt.show()
