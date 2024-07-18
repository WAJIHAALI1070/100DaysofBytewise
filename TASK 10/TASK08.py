#Feature Selection in the Diabetes Dataset
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_diabetes
from sklearn.feature_selection import SelectKBest, mutual_info_regression, RFE
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# Load the Diabetes dataset
diabetes_data = load_diabetes()
X = pd.DataFrame(diabetes_data.data, columns=diabetes_data.feature_names)
y = pd.Series(diabetes_data.target, name='target')

# Check the first few rows and data structure
print(X.head())
print(X.info())

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Perform correlation analysis (optional)
correlation_matrix = X_train.corr()
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
plt.title('Correlation Matrix')
plt.show()

# Perform mutual information feature selection
mi_scores = mutual_info_regression(X_train, y_train)
mi_scores = pd.Series(mi_scores, name='MI Scores', index=X.columns)
mi_scores = mi_scores.sort_values(ascending=False)
print(mi_scores)

# Perform Recursive Feature Elimination (RFE)
# Example with Linear Regression
lr = LinearRegression()
rfe_lr = RFE(estimator=lr, n_features_to_select=5, step=1)
rfe_lr.fit(X_train, y_train)

print("Selected features by RFE with Linear Regression:")
print(X.columns[rfe_lr.support_])

# Example with Random Forest Regressor
rf = RandomForestRegressor(n_estimators=100, random_state=42)
rfe_rf = RFE(estimator=rf, n_features_to_select=5, step=1)
rfe_rf.fit(X_train, y_train)

print("Selected features by RFE with Random Forest:")
print(X.columns[rfe_rf.support_])

# Evaluate selected features using a model (e.g., Linear Regression)
selected_features = X.columns[rfe_lr.support_]
lr.fit(X_train[selected_features], y_train)
y_pred = lr.predict(X_test[selected_features])
mse = mean_squared_error(y_test, y_pred)
print(f"Mean Squared Error using selected features: {mse}")

# Visualize important features
plt.figure(figsize=(10, 6))
plt.barh(selected_features, lr.coef_)
plt.xlabel('Coefficient Magnitude')
plt.title('Importance of Selected Features')
plt.show()
