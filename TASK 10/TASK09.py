#Dealing with Imbalanced Data in the Credit Card Fraud Detection Dataset
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE, ADASYN
from imblearn.under_sampling import RandomUnderSampler
from collections import Counter

# Step 1: Load the dataset
file_path = r'C:\Users\pc\Downloads\100DaysOfBytewise_Task10\creditcard.csv'
df = pd.read_csv(file_path)

# Display the first few rows of the dataset to understand its structure
print("Dataset preview:")
print(df.head())

# Step 2: Separate features and target variable
X = df.drop('Class', axis=1)  # Features
y = df['Class']  # Target variable

# Step 3: Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 4: Resample the training data using SMOTE (Synthetic Minority Over-sampling Technique)
print("Original class distribution in training set:", Counter(y_train))

smote = SMOTE(random_state=42)
X_resampled_smote, y_resampled_smote = smote.fit_resample(X_train, y_train)

print("Resampled class distribution using SMOTE:", Counter(y_resampled_smote))

# Step 5: Resample the training data using ADASYN (Adaptive Synthetic Sampling)
adasyn = ADASYN(random_state=42)
X_resampled_adasyn, y_resampled_adasyn = adasyn.fit_resample(X_train, y_train)

print("Resampled class distribution using ADASYN:", Counter(y_resampled_adasyn))

# Step 6: Undersample the majority class using RandomUnderSampler
undersampler = RandomUnderSampler(random_state=42)
X_resampled_under, y_resampled_under = undersampler.fit_resample(X_train, y_train)

print("Resampled class distribution using RandomUnderSampler:", Counter(y_resampled_under))
