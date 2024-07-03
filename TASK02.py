#Exercise: Calculate the variance and standard deviation of the petal widths in the Iris dataset.
import pandas as pd
import numpy as np
import seaborn as sns

# Load the Iris dataset from Seaborn
iris = sns.load_dataset('iris')

# Calculate variance and standard deviation of petal widths
variance_petal_width = np.var(iris['petal_width'], ddof=0)  # Population variance
std_deviation_petal_width = np.std(iris['petal_width'], ddof=0)  # Population standard deviation

print(f"Population Variance of petal widths: {variance_petal_width:.4f}")
print(f"Population Standard deviation of petal widths: {std_deviation_petal_width:.4f}")
