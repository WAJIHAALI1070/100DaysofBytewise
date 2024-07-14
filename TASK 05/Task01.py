# Import necessary libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

#Checking if all the libraries are imported successfully
print("All libraries imported successfully!")

#since the Boston dataset was removed for ethical reasons, California housing is being used.
housing = fetch_california_housing()
#fetch california housing data set from sklearn.datasets

#now we convert the dataset imported to pandas DataFrame
data = pd.DataFrame(housing.data, columns= housing.feature_names)
#housing.data contains data as a Numpy Array and housing.feature_names has the names of the features as a list
#pd.DataFrame converts the NumPy array into a pandas DataFrame and assigns the column names to the DataFrame using the housing.feature_names list

#now to add a target variable (median house value) to the DataFrame
data['MedHouseVal'] = housing.target

#now to see what the data is and what data is loaded lets have a look at the data itself
print(data.head())

#now to define features(independent variables) and targets(dependent variables)
#define feature(X) i.e independent variable
X = data.drop('MedHouseVal', axis=1)

#and now to define target(Y) i.e dependent variable
Y= data['MedHouseVal']

#Features such as MedInc, HouseAge, AveRooms, etc., are chosen as independent variables (X) because they are used to predict the median house value (MedHouseVal), which is the dependent variable (Y).
#now in linear regression y is what we aim to model using X and therefore we must at first split the data into test and training datasets
# Split the data into training and testing sets
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
#The training set is used to train your machine learning model. It contains a known output (target variable y) and the corresponding features (X) used to learn the model.
#The test set is used to evaluate the performance of the model after it has been trained.
#test_size = 0.2 means 20% of the data will be used as a test set

#now to create a linear regression model
model = LinearRegression()

#once the model is created we train the model. It will be trained on X and Y
model.fit(X_train, Y_train)

#now to understand coefficeints and intercetp in linear regression
#coefficients:  relationship between each independent variable (X) and the dependent variable (y)
#intercept: the value of y when all independent variables (X) are set to zero, shifts regression line up or down
# Print the model's coefficients and intercept
print("Coefficients:", model.coef_)
print("Intercept:", model.intercept_)

#once the model is trained, we predict i.e test the model on testing data
Y_pred = model.predict(X_test)

# Calculate and print the mean squared error
mse = mean_squared_error(Y_test, Y_pred)
print("Mean Squared Error:", mse)

# Visualize the regression line and data points
plt.figure(figsize=(10, 6))
plt.scatter(Y_test, Y_pred, edgecolors=(0, 0, 0))
plt.plot([Y_test.min(), Y_test.max()], [Y_test.min(), Y_test.max()], 'k--', lw=3)
plt.xlabel('Actual')
plt.ylabel('Predicted')
plt.title('Actual vs Predicted Housing Prices')
plt.show()