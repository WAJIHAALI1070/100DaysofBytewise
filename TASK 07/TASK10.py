#Exercise: Combine Matplotlib and Seaborn to create a complex visualization, such as overlaying a KDE plot on a histogram.
import seaborn as sns
import matplotlib.pyplot as plt

# Load a built-in dataset from Seaborn
# Let's use the 'iris' dataset as an example
iris = sns.load_dataset('iris')

# Set up the figure and axis
fig, ax = plt.subplots()

# Create a histogram with KDE overlay using Seaborn
sns.histplot(iris['sepal_length'], bins=20, kde=True, ax=ax, color='lightblue', edgecolor='black')
sns.kdeplot(iris['sepal_length'], color='red', ax=ax)

# Customize labels and title
ax.set_xlabel('Sepal Length (cm)')
ax.set_ylabel('Frequency')
ax.set_title('Histogram with KDE Overlay')

# Display the plot
plt.show()
