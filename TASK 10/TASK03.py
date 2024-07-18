#Scaling Features in the Wine Quality Dataset
# scaling_features_wine_quality.py

# Import necessary libraries
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import matplotlib.pyplot as plt

# Load the Wine Quality dataset
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv"
wine_df = pd.read_csv(url, sep=';')

# Display the first few rows of the dataset
print("Initial Dataset:\n", wine_df.head(), "\n")

# Separate features (attributes) and target (quality)
X = wine_df.drop('quality', axis=1)
y = wine_df['quality']

# Apply Standardization
scaler_standard = StandardScaler()
X_standardized = scaler_standard.fit_transform(X)

# Apply Normalization (Min-Max scaling)
scaler_minmax = MinMaxScaler()
X_normalized = scaler_minmax.fit_transform(X)

# Convert the scaled arrays back to DataFrames for visualization
X_standardized_df = pd.DataFrame(X_standardized, columns=X.columns)
X_normalized_df = pd.DataFrame(X_normalized, columns=X.columns)

# Plot histograms to visualize the effect of scaling on the distribution of data
fig, axs = plt.subplots(2, 2, figsize=(12, 10))

axs[0, 0].set_title('Before Scaling (Original)')
for col in X.columns:
    axs[0, 0].hist(X[col], bins=20, alpha=0.7, label=col)
axs[0, 0].legend()

axs[0, 1].set_title('After Standardization')
for col in X_standardized_df.columns:
    axs[0, 1].hist(X_standardized_df[col], bins=20, alpha=0.7, label=col)
axs[0, 1].legend()

axs[1, 0].set_title('After Normalization')
for col in X_normalized_df.columns:
    axs[1, 0].hist(X_normalized_df[col], bins=20, alpha=0.7, label=col)
axs[1, 0].legend()

# Plot original and scaled data distributions for one feature (e.g., alcohol)
feature = 'alcohol'
axs[1, 1].set_title(f'Distribution of {feature}')
axs[1, 1].hist(X[feature], bins=20, alpha=0.5, label='Original')
axs[1, 1].hist(X_standardized_df[feature], bins=20, alpha=0.7, label='Standardized')
axs[1, 1].hist(X_normalized_df[feature], bins=20, alpha=0.7, label='Normalized')
axs[1, 1].legend()

plt.tight_layout()
plt.show()
