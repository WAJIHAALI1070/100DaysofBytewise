#Exercise: Calculate and interpret the covariance and correlation between sepal length and sepal width.
import seaborn as sns
import numpy as np

# Load the Iris dataset from Seaborn
iris = sns.load_dataset('iris')

# Extract sepal length and sepal width data
sepal_length = iris['sepal_length']
sepal_width = iris['sepal_width']

# Calculate covariance matrix
covariance_matrix = np.cov(sepal_length, sepal_width)

# Extract covariance between sepal length and sepal width
covariance_sepal = covariance_matrix[0, 1]

# Calculate correlation matrix
correlation_matrix = np.corrcoef(sepal_length, sepal_width)

# Extract correlation between sepal length and sepal width
correlation_sepal = correlation_matrix[0, 1]

# Print covariance and correlation
print(f"Covariance between sepal length and sepal width: {covariance_sepal:.4f}")
print(f"Correlation between sepal length and sepal width: {correlation_sepal:.4f}")

# Interpret correlation
if correlation_sepal > 0:
    print("There is a positive correlation between sepal length and sepal width.")
elif correlation_sepal < 0:
    print("There is a negative correlation between sepal length and sepal width.")
else:
    print("There is no linear correlation between sepal length and sepal width.")
