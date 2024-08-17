#Customer Churn Prediction
#Muhammad Huzaifa ---------mhuzaifa287e@gmail.com
#Wajiha Ali----------------wajihaali1070@gmail.com

#importing libraries
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

print("Imported Libraries Successfully")

# Load the dataset
file_path = r"C:\Users\pc\Downloads\Customer Churn Prediction\Customer Churn Prediction\WA_Fn-UseC_-Telco-Customer-Churn.csv"

print("Loaded the dataset")
# Read the CSV file
df = pd.read_csv(file_path)

# Display the first few rows of the DataFrame
print(df.head())

# Drop customerID column as it is not needed for prediction
df.drop('customerID', axis=1, inplace=True)

# Handle missing values
print("Handling missing values")
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
df.fillna(df['TotalCharges'].mean(), inplace=True)

# Encode categorical variables
label_encoders = {}
for column in df.select_dtypes(include=['object']).columns:
    le = LabelEncoder()
    df[column] = le.fit_transform(df[column])
    label_encoders[column] = le

# Separate features and target
print("Separate features and target")
X = df.drop('Churn', axis=1)
y = df['Churn']

# Split the dataset into training and testing sets
print("Splitting the dataset into training and testing sets")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Standardize the features
print("Standardize the features")
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Define the hyperparameter grid for GridSearchCV
print("Define the hyperparameter grid for GridSearchCV")
param_grid = {
    'n_estimators': [100, 200, 300],
    'learning_rate': [0.01, 0.05, 0.1],
    'max_depth': [3, 4, 5],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

# Initialize GradientBoostingClassifier and GridSearchCV
gbm = GradientBoostingClassifier(random_state=42)
grid_search = GridSearchCV(estimator=gbm, param_grid=param_grid, cv=5, n_jobs=-1, verbose=2)

# Fit GridSearchCV to the training data
print("Performing Grid Search")
grid_search.fit(X_train, y_train)

# Get the best model from GridSearchCV
best_model = grid_search.best_estimator_

# Train the best model
print("Training the best model")
best_model.fit(X_train, y_train)

# Save the trained model, scaler, and label encoders
print("Saving the model, scaler, and label encoders")
joblib.dump(best_model, 'churn_model.pkl')
joblib.dump(scaler, 'scaler.pkl')
joblib.dump(label_encoders, 'label_encoders.pkl')

# Evaluate the model
print("Evaluating the model")
y_pred = best_model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)
report = classification_report(y_test, y_pred)

# Output the evaluation metrics
print("Accuracy: {:.2f}%".format(accuracy * 100))
print("Confusion Matrix:")
print(cm)
print("Classification Report:")
print(report)

# Plot Confusion Matrix
plt.figure(figsize=(10, 7))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Not Churn', 'Churn'], yticklabels=['Not Churn', 'Churn'])
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()
