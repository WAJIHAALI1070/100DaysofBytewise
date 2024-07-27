# Machine Learning Tasks on Adult Income Dataset

This repository contains Python scripts (`.py`) or Jupyter Notebooks (`.ipynb`) for analyzing the Adult Income dataset. The tasks cover various machine learning techniques and evaluations using Python libraries.

## Task Overview

### Task 01: Applying Cross-Validation to Random Forest Classifier
**Exercise:** Implement a random forest classifier and evaluate the model using k-fold cross-validation. Analyze the cross-validation scores to assess model stability.

- **Objective:** Assess model stability using k-fold cross-validation.
- **Methods:** Random Forest Classifier, Cross-Validation.
- **Outcome:** Cross-validation scores and their mean.

### Task 02: Investigating Overfitting and Underfitting in Gradient Boosting Machines
**Exercise:** Train a gradient boosting classifier with varying numbers of estimators and learning rates. Evaluate the model for overfitting and underfitting by comparing training and validation performance.

- **Objective:** Identify signs of overfitting and underfitting.
- **Methods:** Gradient Boosting Classifier with different hyperparameters.
- **Outcome:** Training and validation scores comparison.

### Task 03: Evaluating Precision, Recall, and F1-Score for Random Forests
**Exercise:** Implement a random forest classifier and calculate precision, recall, and F1-score. Discuss the trade-offs between these metrics and their importance for classification tasks.

- **Objective:** Evaluate the model using precision, recall, and F1-score.
- **Methods:** Random Forest Classifier, Precision, Recall, F1-Score.
- **Outcome:** Metric scores and discussion on trade-offs.

### Task 04: ROC Curve and AUC for Gradient Boosting Classifier
**Exercise:** Implement a gradient boosting classifier and plot the ROC curve. Compute the AUC and interpret how well the model distinguishes between classes.

- **Objective:** Assess the model's ability to distinguish between classes.
- **Methods:** Gradient Boosting Classifier, ROC Curve, AUC.
- **Outcome:** ROC curve plot and AUC score.

### Task 05: Model Performance Comparison with Different Metrics
**Exercise:** Compare the performance of different classifiers (e.g., SVM, random forest, gradient boosting) using cross-validation. Evaluate and compare the models' performance metrics.

- **Objective:** Compare multiple classifiers on performance metrics.
- **Methods:** SVM, Random Forest, Gradient Boosting, Cross-Validation.
- **Outcome:** Performance comparison across different classifiers.

## Dataset Description

The Adult Income dataset is used to predict whether a person makes over $50K a year based on census data. The dataset contains the following columns:
- Age
- Workclass
- Final weight
- Education
- Education-num
- Marital status
- Occupation
- Relationship
- Race
- Sex
- Capital-gain
- Capital-loss
- Hours-per-week
- Native-country
- Income

## Installation and Usage

To run the scripts or notebooks in this repository, you need to have Python installed along with the following libraries:

- pandas
- numpy
- scikit-learn
- matplotlib
- seaborn
- jupyter

You can install the required libraries using pip:
```bash
pip install pandas numpy scikit-learn matplotlib seaborn jupyter

