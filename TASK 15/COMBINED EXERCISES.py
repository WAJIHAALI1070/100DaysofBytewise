# Import necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import precision_score, recall_score, f1_score, roc_curve, auc
from sklearn.svm import SVC
import matplotlib.pyplot as plt

# Load the dataset
url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data'
column_names = ['age', 'workclass', 'fnlwgt', 'education', 'education-num', 'marital-status',
                'occupation', 'relationship', 'race', 'sex', 'capital-gain', 'capital-loss',
                'hours-per-week', 'native-country', 'income']
data = pd.read_csv(url, names=column_names, sep=', ', engine='python')

# Data preprocessing
data = pd.get_dummies(data, drop_first=True)
X = data.drop('income_>50K', axis=1)
y = data['income_>50K']

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Exercise 1: Cross-Validation with Random Forest Classifier
rf_clf = RandomForestClassifier(n_estimators=100, random_state=42)
cv_scores = cross_val_score(rf_clf, X_train, y_train, cv=5)
print(f"Cross-Validation Scores: {cv_scores}")
print(f"Mean Cross-Validation Score: {cv_scores.mean()}")

# Exercise 2: Investigating Overfitting and Underfitting in Gradient Boosting Machines
learning_rates = [0.01, 0.1, 0.2, 0.3]
n_estimators = [50, 100, 200]
for lr in learning_rates:
    for n_est in n_estimators:
        gb_clf = GradientBoostingClassifier(learning_rate=lr, n_estimators=n_est, random_state=42)
        gb_clf.fit(X_train, y_train)
        train_score = gb_clf.score(X_train, y_train)
        test_score = gb_clf.score(X_test, y_test)
        print(f"Learning Rate: {lr}, Estimators: {n_est} -> Train Score: {train_score}, Test Score: {test_score}")

# Exercise 3: Evaluating Precision, Recall, and F1-Score for Random Forests
rf_clf.fit(X_train, y_train)
y_pred = rf_clf.predict(X_test)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
print(f"Precision: {precision}, Recall: {recall}, F1-Score: {f1}")

# Exercise 4: ROC Curve and AUC for Gradient Boosting Classifier
gb_clf = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, random_state=42)
gb_clf.fit(X_train, y_train)
y_pred_proba = gb_clf.predict_proba(X_test)[:, 1]
fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
roc_auc = auc(fpr, tpr)
plt.figure()
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:0.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic')
plt.legend(loc="lower right")
plt.show()

# Exercise 5: Model Performance Comparison with Different Metrics
classifiers = {
    'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
    'Gradient Boosting': GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, random_state=42),
    'SVM': SVC(probability=True, random_state=42)
}

for clf_name, clf in classifiers.items():
    cv_scores = cross_val_score(clf, X_train, y_train, cv=5, scoring='accuracy')
    print(f"{clf_name} Cross-Validation Accuracy: {cv_scores.mean()}")
