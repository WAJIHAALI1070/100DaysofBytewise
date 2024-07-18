#Data Imputation in the Retail Sales Dataset
#CSV file i.e dataset path: C:\Users\pc\Downloads\100DaysOfBytewise_Task10\retail_sales_dataset.csv'

import pandas as pd
from sklearn.impute import KNNImputer
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer

# Load the dataset
df = pd.read_csv(r'C:\Users\pc\Downloads\100DaysOfBytewise_Task10\retail_sales_dataset.csv')

# Check for missing values
print("Missing values before imputation:")
print(df.isnull().sum())

# Separate numeric and non-numeric columns
numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
non_numeric_cols = df.select_dtypes(exclude=['float64', 'int64']).columns

# KNN Imputation for numeric columns
imputer_knn = KNNImputer(n_neighbors=5)
df_numeric_knn_imputed = imputer_knn.fit_transform(df[numeric_cols])
df_numeric_knn_imputed = pd.DataFrame(df_numeric_knn_imputed, columns=numeric_cols)

# MICE (Multiple Imputation by Chained Equations) for numeric columns
imputer_mice = IterativeImputer()
df_numeric_mice_imputed = imputer_mice.fit_transform(df[numeric_cols])
df_numeric_mice_imputed = pd.DataFrame(df_numeric_mice_imputed, columns=numeric_cols)

# Combine the imputed numeric columns with non-numeric columns
df_knn_imputed = pd.concat([df_numeric_knn_imputed, df[non_numeric_cols].reset_index(drop=True)], axis=1)
df_mice_imputed = pd.concat([df_numeric_mice_imputed, df[non_numeric_cols].reset_index(drop=True)], axis=1)

# Check for missing values after imputation
print("Missing values after KNN imputation:")
print(df_knn_imputed.isnull().sum())

print("Missing values after MICE imputation:")
print(df_mice_imputed.isnull().sum())

# Save the imputed datasets if necessary
# df_knn_imputed.to_csv('retail_sales_knn_imputed.csv', index=False)
# df_mice_imputed.to_csv('retail_sales_mice_imputed.csv', index=False)

# Display the first few rows of the imputed datasets
print("First few rows of KNN imputed dataset:")
print(df_knn_imputed.head())

print("First few rows of MICE imputed dataset:")
print(df_mice_imputed.head())
