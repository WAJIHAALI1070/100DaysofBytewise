#Exercise: Create a box plot using Seaborn to show the distribution of values for different categories in a dataset.
import seaborn as sns
import matplotlib.pyplot as plt

# Load a built-in dataset from Seaborn
#using the 'tips' dataset as an example
tips = sns.load_dataset('tips')

# Create a box plot with updated parameters
sns.boxplot(x='day', y='total_bill', hue='day', data=tips, palette='Set3', legend=False)

# Add labels and title
plt.xlabel('Day of the Week')
plt.ylabel('Total Bill ($)')
plt.title('Box Plot of Total Bill by Day')

# Display the plot
plt.show()
