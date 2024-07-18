# Feature Engineering in the Heart Disease Dataset
import pandas as pd

# URL for the Heart Disease dataset from UCI Machine Learning Repository
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data"

# Column names for the dataset
column_names = [
    "age", "sex", "cp", "trestbps", "chol", "fbs", "restecg", "thalach",
    "exang", "oldpeak", "slope", "ca", "thal", "target"
]

# Load the dataset from the URL
df = pd.read_csv(url, header=None, names=column_names, na_values="?")

# Display the first few rows of the dataset
print("First few rows of the original dataset:")
print(df.head())

# Check for missing values
print("Missing values in the dataset:")
print(df.isnull().sum())

# Drop rows with missing values for simplicity (optional: you could also impute them)
df.dropna(inplace=True)

# Create age groups
def create_age_group(age):
    if age < 35:
        return 'Young'
    elif 35 <= age < 55:
        return 'Middle-aged'
    else:
        return 'Old'

df['Age Group'] = df['age'].apply(create_age_group)

# Create cholesterol level categories
def create_cholesterol_level(chol):
    if chol < 200:
        return 'Desirable'
    elif 200 <= chol < 240:
        return 'Borderline high'
    else:
        return 'High'

df['Cholesterol Level'] = df['chol'].apply(create_cholesterol_level)

# Example of combining multiple features into a single feature (Risk Score)
df['Risk Score'] = df['age'] * 0.2 + df['chol'] * 0.3 + df['trestbps'] * 0.5

# Display the first few rows of the new dataset with engineered features
print("First few rows of the dataset with new features:")
print(df.head())
