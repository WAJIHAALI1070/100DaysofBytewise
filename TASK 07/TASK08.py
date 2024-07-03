#Exercise: Customize the appearance of a Seaborn plot by changing the color palette, adding titles, and modifying axis labels.
import seaborn as sns
import matplotlib.pyplot as plt

# Load a built-in dataset from Seaborn
#use the 'iris' dataset as an example
iris = sns.load_dataset('iris')

# Create a scatter plot
sns.scatterplot(x='sepal_length', y='sepal_width', data=iris, hue='species', palette='Set2')

# Customize appearance
plt.title('Sepal Length vs Sepal Width')
plt.xlabel('Sepal Length (cm)')
plt.ylabel('Sepal Width (cm)')

# Display the plot
plt.legend(title='Species', loc='upper left')  # Customize legend
plt.tight_layout()
plt.show()
