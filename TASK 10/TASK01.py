#Handling Missing Data in Titanic Dataset
# handling_missing_data_titanic.py

import pandas as pd
import numpy as np

# Load the Titanic dataset
url = "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"
titanic_df = pd.read_csv(url)

# Display the first few rows of the dataset
print("Initial Dataset:\n", titanic_df.head(), "\n")

# Identify missing values
missing_values = titanic_df.isnull().sum()
print("Missing Values:\n", missing_values, "\n")

# Handle missing values

# 1. Mean/Median Imputation
titanic_df_mean_impute = titanic_df.copy()
titanic_df_mean_impute['Age'].fillna(titanic_df_mean_impute['Age'].mean(), inplace=True)
titanic_df_median_impute = titanic_df.copy()
titanic_df_median_impute['Age'].fillna(titanic_df_median_impute['Age'].median(), inplace=True)

# 2. Mode Imputation
titanic_df_mode_impute = titanic_df.copy()
titanic_df_mode_impute['Embarked'].fillna(titanic_df_mode_impute['Embarked'].mode()[0], inplace=True)

# 3. Dropping Rows/Columns
titanic_df_drop_rows = titanic_df.dropna()  # Drop rows with any missing values
titanic_df_drop_columns = titanic_df.drop(columns=['Cabin'])  # Drop columns with missing values

# Display the datasets after handling missing values
print("Dataset after Mean Imputation:\n", titanic_df_mean_impute.head(), "\n")
print("Dataset after Median Imputation:\n", titanic_df_median_impute.head(), "\n")
print("Dataset after Mode Imputation:\n", titanic_df_mode_impute.head(), "\n")
print("Dataset after Dropping Rows:\n", titanic_df_drop_rows.head(), "\n")
print("Dataset after Dropping Columns:\n", titanic_df_drop_columns.head(), "\n")
