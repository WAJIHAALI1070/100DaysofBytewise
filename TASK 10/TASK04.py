#Handling Outliers in the Boston Housing Dataset
# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.datasets import fetch_california_housing

# Load the California Housing dataset
california = fetch_california_housing()
california_df = pd.DataFrame(california.data, columns=california.feature_names)
california_df['MEDV'] = california.target  # Add the target variable (median house value)

# Display the first few rows of the dataset
print("Initial Dataset:\n", california_df.head(), "\n")

# Define a function to identify outliers using Z-score
def detect_outliers_zscore(data, threshold=3):
    z_scores = np.abs(stats.zscore(data))
    return np.where(z_scores > threshold)

# Define a function to identify outliers using IQR (Interquartile Range)
def detect_outliers_iqr(data):
    Q1 = data.quantile(0.25)
    Q3 = data.quantile(0.75)
    IQR = Q3 - Q1
    return data[((data < (Q1 - 1.5 * IQR)) | (data > (Q3 + 1.5 * IQR))).any(axis=1)]

# Detect and handle outliers using Z-score
outliers_zscore = detect_outliers_zscore(california_df.drop(columns=['MEDV']))
print("Outliers detected using Z-score:")
print(outliers_zscore)

# Detect and handle outliers using IQR
outliers_iqr = detect_outliers_iqr(california_df.drop(columns=['MEDV']))
print("\nOutliers detected using IQR:")
print(outliers_iqr)

# Visualize outliers using boxplots
plt.figure(figsize=(12, 6))
sns.boxplot(data=california_df.drop(columns=['MEDV']), orient="h", palette="Set2")
plt.title("Boxplot of California Housing Dataset Features")
plt.xlabel("Values")
plt.show()
