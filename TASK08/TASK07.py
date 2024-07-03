#Exercise: Calculate and plot the probability density function (PDF) for sepal width.
import seaborn as sns
import matplotlib.pyplot as plt

# Load the Iris dataset from Seaborn
iris = sns.load_dataset('iris')

# Extract sepal width data
sepal_width = iris['sepal_width']

# Plotting
plt.figure(figsize=(8, 6))
sns.kdeplot(sepal_width, fill=True, color='b')
plt.title('Probability Density Function (PDF) of Sepal Width')
plt.xlabel('Sepal Width')
plt.ylabel('Density')
plt.grid(True)
plt.show()

