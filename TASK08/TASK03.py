#Exercise: Create a summary table that includes the mean, median, variance, and standard deviation for all numerical features in the dataset.
import numpy as np
import pandas as pd
import seaborn as sns

# Load the Iris dataset from Seaborn
iris = sns.load_dataset('iris')

# Select only numeric columns for variance calculation
numeric_columns = iris.select_dtypes(include=[np.number])

# Calculate mean, median, variance, and standard deviation for all numerical features
summary_stats = numeric_columns.describe().loc[['mean', '50%', 'std']].transpose()

# Calculate variance separately and add it to the summary table
variance = numeric_columns.var().to_frame().transpose()
variance.index = ['var']
summary_stats = pd.concat([summary_stats, variance])

print("Summary statistics for numerical features in the Iris dataset:")
print(summary_stats)
