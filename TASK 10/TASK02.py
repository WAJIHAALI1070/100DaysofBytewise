#Encoding Categorical Variables in a Car Evaluation Dataset
import pandas as pd
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

# Load the Car Evaluation dataset
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/car/car.data"
columns = ['buying', 'maint', 'doors', 'persons', 'lug_boot', 'safety', 'class']
car_df = pd.read_csv(url, names=columns)

# Display the first few rows of the dataset
print("Initial Dataset:\n", car_df.head(), "\n")

# Encode categorical variables using Label Encoding
label_encoder = LabelEncoder()
car_df_label_encoded = car_df.copy()

for column in car_df_label_encoded.columns:
    car_df_label_encoded[column] = label_encoder.fit_transform(car_df_label_encoded[column])

print("Dataset after Label Encoding:\n", car_df_label_encoded.head(), "\n")

# Encode categorical variables using One-Hot Encoding
one_hot_encoder = OneHotEncoder(sparse_output=False)
car_df_one_hot_encoded = pd.get_dummies(car_df.copy(), columns=car_df.columns)

print("Dataset after One-Hot Encoding:\n", car_df_one_hot_encoded.head(), "\n")#