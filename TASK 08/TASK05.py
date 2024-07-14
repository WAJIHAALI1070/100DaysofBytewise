#Exercise: Plot the probability distribution of sepal lengths using a histogram.
import seaborn as sns
import matplotlib.pyplot as plt

# Load the Iris dataset from Seaborn
iris = sns.load_dataset('iris')

# Extract sepal length data
sepal_length = iris['sepal_length']

# Plotting
plt.figure(figsize=(8, 6))
sns.histplot(sepal_length, kde=True, stat='probability', bins=20, color='skyblue')
plt.title('Probability Distribution of Sepal Lengths')
plt.xlabel('Sepal Length')
plt.ylabel('Probability Density')
plt.grid(True)
plt.show()
