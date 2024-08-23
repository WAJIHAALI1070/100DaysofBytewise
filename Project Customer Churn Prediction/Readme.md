# Customer Churn Prediction


This project aims to predict customer churn for a telecom company using a machine learning model. The project involves data preprocessing, visualization, model training, and evaluation, followed by saving the trained model for future predictions. The dataset used for this project is the Telco Customer Churn dataset.

## Table of Contents

- [Introduction](#introduction)
- [Project Overview](#project-overview)
- [Data Preprocessing](#data-preprocessing)
- [Data Visualization](#data-visualization)
- [Model Training and Evaluation](#model-training-and-evaluation)
- [Model Saving](#model-saving)
- [Dependencies](#dependencies)
- [How to Run](#how-to-run)
- [Authors](#authors)

#Introduction

Customer churn prediction is a vital aspect of customer relationship management (CRM) and is used by businesses to retain valuable customers. In this project, we predict customer churn using a machine learning model based on the Telco Customer Churn dataset. The dataset contains customer demographics, account information, and service usage details.

## Project Overview

1. **Data Collection**: The dataset is loaded from a CSV file.
2. **Data Preprocessing**: Handling missing values, converting categorical variables to numerical, and standardizing numerical features.
3. **Data Visualization**: Visualizing relationships between features and the target variable (churn).
4. **Model Training and Evaluation**: A RandomForestClassifier is trained and evaluated on the preprocessed data.
5. **Model Saving**: The trained model is saved as a .pkl file for future use.

## Data Preprocessing

- Missing values are handled by imputing the median for numerical columns and the mode for categorical columns.
- Categorical variables are converted to numerical using one-hot encoding.
- Numerical features are standardized to bring them to a comparable scale.

## Data Visualization

Several visualizations are used to explore the data:
- Correlation matrix to identify relationships between features.
- Churn distribution to understand the class imbalance.
- Box plots to compare tenure and monthly charges against churn.
- Pair plots to visualize the relationships among important features.

## Model Training and Evaluation

- The data is split into training and testing sets.
- A RandomForestClassifier is trained on the training set.
- The model is evaluated on the test set using accuracy, confusion matrix, and classification report.

## Model Saving

- The trained model is saved as `churn_model.pkl` using the `joblib` library.

## Dependencies

- pandas
- numpy
- scikit-learn
- matplotlib
- seaborn
- joblib

## How to Run

1. Clone the repository.
2. Install the dependencies using the following command:
   ```bash
   pip install -r requirements.txt
   ```
3. Run the Jupyter notebook or Python script.
## Authors
- **Muhammad Huzaifa** - mhuzaifa287e@gmail.com
- **Wajiha Ali** - wajihaali1070@gmail.com
