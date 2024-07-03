#Exercise: Calculate the mean, median, and mode of the sepal lengths in the Iris dataset.
import pandas as pd
from scipy import stats
import seaborn as sns

# Load the Iris dataset from Seaborn
iris = sns.load_dataset('iris')

# Calculate mean, median, and mode of sepal lengths
mean_sepal_length = iris['sepal_length'].mean()
median_sepal_length = iris['sepal_length'].median()
mode_sepal_length = iris['sepal_length'].mode()[0]  # mode()[0] gives the first mode in case of multimodal distribution

print(f"Mean sepal length: {mean_sepal_length:.2f}")
print(f"Median sepal length: {median_sepal_length}")
print(f"Mode sepal length: {mode_sepal_length}")
