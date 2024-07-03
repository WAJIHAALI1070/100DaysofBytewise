#Exercise: Plot a heatmap using Seaborn to visualize the correlation matrix of a dataset.
import seaborn as sns
import matplotlib.pyplot as plt

# Load a built-in dataset from Seaborn
# loading iris data set to be used as an example
iris = sns.load_dataset('iris')

# Drop the 'species' column before computing the correlation matrix
numeric_columns = iris.drop(columns='species')

# Compute the correlation matrix
correlation_matrix = numeric_columns.corr()

# Create a heatmap
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1)

# Add title
plt.title('Correlation Matrix Heatmap')

# Display the plot
plt.show()
