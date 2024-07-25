import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_curve, auc
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score, KFold
import matplotlib.pyplot as plt

# Load the Titanic dataset
titanic_url = 'https://www.kaggle.com/datasets/yasserh/titanic-dataset'
titanic_df = pd.read_csv(titanic_url)

# Preprocessing
def preprocess_data(df):
    # Dropping unnecessary columns
    df = df.drop(columns=['PassengerId', 'Name', 'Ticket', 'Cabin'])
    
    # Defining numerical and categorical features
    numerical_features = ['Age', 'SibSp', 'Parch', 'Fare']
    categorical_features = ['Sex', 'Embarked']
    
    # Numerical pipeline
    numerical_pipeline = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    
    # Categorical pipeline
    categorical_pipeline = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])
    
    # Combining pipelines
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numerical_pipeline, numerical_features),
            ('cat', categorical_pipeline, categorical_features)
        ])
    
    # Applying preprocessing
    df_preprocessed = preprocessor.fit_transform(df)
    
    return df_preprocessed

# Applying preprocessing to features
X = preprocess_data(titanic_df.drop(columns=['Survived']))
y = titanic_df['Survived']

# Splitting data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Logistic Regression with Cross-Validation
log_reg = LogisticRegression(max_iter=1000)
kf = KFold(n_splits=5, shuffle=True, random_state=42)

# Cross-validation scores
cv_scores = cross_val_score(log_reg, X_train, y_train, cv=kf, scoring='accuracy')
print(f'Cross-validation accuracy scores: {cv_scores}')
print(f'Mean cross-validation accuracy: {cv_scores.mean()}')

# Single train-test split evaluation
log_reg.fit(X_train, y_train)
y_pred = log_reg.predict(X_test)
single_split_accuracy = accuracy_score(y_test, y_pred)
print(f'Single train-test split accuracy: {single_split_accuracy}')

# Analyzing Overfitting and Underfitting in Decision Trees
train_accuracies = []
val_accuracies = []
depths = range(1, 21)

for depth in depths:
    tree = DecisionTreeClassifier(max_depth=depth, random_state=42)
    tree.fit(X_train, y_train)
    
    train_acc = accuracy_score(y_train, tree.predict(X_train))
    val_acc = accuracy_score(y_test, tree.predict(X_test))
    
    train_accuracies.append(train_acc)
    val_accuracies.append(val_acc)

# Plotting the accuracies
plt.figure(figsize=(10, 6))
plt.plot(depths, train_accuracies, label='Training Accuracy')
plt.plot(depths, val_accuracies, label='Validation Accuracy')
plt.xlabel('Tree Depth')
plt.ylabel('Accuracy')
plt.legend()
plt.title('Training and Validation Accuracies for Decision Trees with Varying Depths')
plt.show()

# Calculating Precision, Recall, and F1-Score for Logistic Regression
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print(f'Precision: {precision}')
print(f'Recall: {recall}')
print(f'F1-Score: {f1}')

# ROC Curve Analysis for Decision Trees
tree = DecisionTreeClassifier(max_depth=5, random_state=42)
tree.fit(X_train, y_train)

y_proba = tree.predict_proba(X_test)[:, 1]
fpr, tpr, thresholds = roc_curve(y_test, y_proba)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='blue', lw=2, label=f'ROC Curve (AUC = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='grey', lw=1, linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve for Decision Tree')
plt.legend(loc='lower right')
plt.show()

# Comparing Model Performance with and without Cross-Validation
def evaluate_model(model, X_train, X_test, y_train, y_test):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    
    return accuracy, precision, recall, f1

# Logistic Regression
log_reg_metrics = evaluate_model(log_reg, X_train, X_test, y_train, y_test)
print(f'Logistic Regression (without cross-validation): {log_reg_metrics}')

cv_log_reg_metrics = cross_val_score(log_reg, X_train, y_train, cv=kf, scoring='accuracy')
print(f'Logistic Regression (with cross-validation): Mean accuracy = {cv_log_reg_metrics.mean()}')

# Decision Tree
tree_metrics = evaluate_model(tree, X_train, X_test, y_train, y_test)
print(f'Decision Tree (without cross-validation): {tree_metrics}')

cv_tree_metrics = cross_val_score(tree, X_train, y_train, cv=kf, scoring='accuracy')
print(f'Decision Tree (with cross-validation): Mean accuracy = {cv_tree_metrics.mean()}')
