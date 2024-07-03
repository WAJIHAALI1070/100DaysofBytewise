#Exercise: Create a violin plot using Seaborn to visualize the distribution of a dataset across different categories.
import seaborn as sns
import matplotlib.pyplot as plt

# Load a built-in dataset from Seaborn
#use the 'iris' dataset as an example
iris = sns.load_dataset('iris')

# Create a violin plot with updated parameters
sns.violinplot(x='species', y='sepal_length', data=iris, hue='species', palette='Set2', legend=False)

# Add labels and title
plt.xlabel('Species')
plt.ylabel('Sepal Length (cm)')
plt.title('Violin Plot of Sepal Length by Species')

# Display the plot
plt.show()

