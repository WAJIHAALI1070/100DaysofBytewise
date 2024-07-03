#Exercise: Load a dataset using Seaborn's built-in dataset functions and create a pairplot to visualize the relationships between all pairs of features.
import seaborn as sns
import matplotlib.pyplot as plt

# Load a built-in dataset from Seaborn
# using Iris data set as an example
iris = sns.load_dataset('iris')

# Create a pairplot with color differentiation based on 'species'
sns.pairplot(iris, hue='species', palette='husl')

# Display the plot
plt.show()

