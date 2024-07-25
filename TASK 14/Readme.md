# Task 14: Machine Learning

This task focuses on implementing and evaluating machine learning models using the Titanic dataset. It covers logistic regression and decision trees with an emphasis on cross-validation, performance metrics, and analyzing overfitting and underfitting.

## Table of Contents
- [Dataset](#dataset)
- [Tasks](#tasks)
  - [Evaluating Logistic Regression with Cross-Validation](#evaluating-logistic-regression-with-cross-validation)
  - [Analyzing Overfitting and Underfitting in Decision Trees](#analyzing-overfitting-and-underfitting-in-decision-trees)
  - [Calculating Precision, Recall, and F1-Score for Logistic Regression](#calculating-precision-recall-and-f1-score-for-logistic-regression)
  - [ROC Curve Analysis for Decision Trees](#roc-curve-analysis-for-decision-trees)
  - [Comparing Model Performance with and without Cross-Validation](#comparing-model-performance-with-and-without-cross-validation)
- [Results and Analysis](#results-and-analysis)
- [Conclusion](#conclusion)
- [Repository Structure](#repository-structure)
- [References](#references)

## Dataset
The Titanic dataset is used for this task. You can download it from [Kaggle](https://www.kaggle.com/datasets/yasserh/titanic-dataset).

## Tasks

### Evaluating Logistic Regression with Cross-Validation
- **Objective:** Implement logistic regression and evaluate the model using k-fold cross-validation.
- **Steps:**
  1. Load and preprocess the data.
  2. Implement logistic regression.
  3. Evaluate the model using k-fold cross-validation.
  4. Compare cross-validation scores with a single train-test split evaluation.

### Analyzing Overfitting and Underfitting in Decision Trees
- **Objective:** Train a decision tree classifier with varying depths to analyze overfitting and underfitting.
- **Steps:**
  1. Train decision trees with different depths.
  2. Plot training and validation accuracies to visualize the effects.

### Calculating Precision, Recall, and F1-Score for Logistic Regression
- **Objective:** Implement logistic regression and calculate precision, recall, and F1-score for the model.
- **Steps:**
  1. Implement logistic regression.
  2. Calculate precision, recall, and F1-score.
  3. Discuss how these metrics provide insights into model performance in your week article.

### ROC Curve Analysis for Decision Trees
- **Objective:** Implement a decision tree classifier and plot the ROC curve.
- **Steps:**
  1. Implement a decision tree classifier.
  2. Plot the ROC curve.
  3. Compute the AUC (Area Under the Curve) and interpret the results.

### Comparing Model Performance with and without Cross-Validation
- **Objective:** Train logistic regression and decision tree models with and without cross-validation. Compare their performance metrics, including accuracy, precision, and recall.
- **Steps:**
  1. Train logistic regression and decision tree models with cross-validation.
  2. Train logistic regression and decision tree models without cross-validation.
  3. Compare performance metrics.

## Results and Analysis
- **Logistic Regression with Cross-Validation:** [Include results and analysis]
- **Decision Trees Overfitting and Underfitting Analysis:** [Include results and analysis]
- **Precision, Recall, and F1-Score for Logistic Regression:** [Include results and analysis]
- **ROC Curve Analysis for Decision Trees:** [Include results and analysis]
- **Model Performance Comparison:** [Include results and analysis]

## Conclusion
In Task 14, we explored various aspects of machine learning using the Titanic dataset, focusing on logistic regression and decision trees. The k-fold cross-validation for logistic regression provided a more robust evaluation compared to a single train-test split, highlighting the importance of cross-validation in preventing overfitting and ensuring model generalization.

The analysis of decision trees with varying depths demonstrated the trade-offs between model complexity and performance. Shallow trees exhibited underfitting, while overly deep trees tended to overfit the training data, emphasizing the need for careful tuning of hyperparameters.

Calculating precision, recall, and F1-score for logistic regression offered deeper insights into the model's performance, particularly in handling imbalanced data. These metrics provided a more comprehensive evaluation compared to accuracy alone.

The ROC curve and AUC analysis for decision trees further illustrated the model's ability to distinguish between classes, with AUC serving as a valuable metric for overall performance.

Comparing models with and without cross-validation highlighted the benefits of using cross-validation to achieve more reliable and unbiased performance metrics. Overall, this task reinforced the importance of thorough model evaluation and the careful balance between underfitting and overfitting in machine learning models.

## Repository Structure
The repository is organized as follows:

- **Task14/**: This directory contains all scripts related to Task 14.
  - `Titanic_Logistic_Regression_CrossValidation.py`: Script for logistic regression with cross-validation.
  - `Decision_Trees_Overfitting_Underfitting.py`: Script for analyzing overfitting and underfitting in decision trees.
  - `Logistic_Regression_Precision_Recall_F1.py`: Script for calculating precision, recall, and F1-score for logistic regression.
  - `Decision_Trees_ROC_Curve.py`: Script for ROC curve analysis for decision trees.
  - `Model_Performance_Comparison.py`: Script for comparing model performance with and without cross-validation.
  - `readme.md`: This readme file.
  
- **datasets/**: This directory contains the dataset used in this task.
  - `titanic.csv`: Titanic dataset file.

## References
- [Titanic Dataset on Kaggle](https://www.kaggle.com/datasets/yasserh/titanic-dataset)
- [Scikit-Learn Documentation](https://scikit-learn.org/stable/documentation.html)



By completing these exercises, we gained practical experience in implementing, evaluating, and tuning machine learning models, preparing us for more advanced challenges in future tasks.
## References
- [Titanic Dataset on Kaggle](https://www.kaggle.com/datasets/yasserh/titanic-dataset)
- [Scikit-Learn Documentation](https://scikit-learn.org/stable/documentation.html)
