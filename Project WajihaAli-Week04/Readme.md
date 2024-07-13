# Predicting Wine Quality Based on Chemical Properties

## Overview
This project focuses on predicting wine quality based on its chemical properties using machine learning techniques. It employs linear regression to model the relationship between various chemical features and wine quality ratings.

## About the Author
Wajiha Ali is a passionate data enthusiast and electrical engineering student with a keen interest in machine learning and data science. Connect with her on [GitHub](https://github.com/WAJIHAALI1070) for more projects and collaborations.
## Additional Resources

- **Medium Article**: [Predicting Wine Quality Using Machine Learning - A Step-by-Step Guide](https://medium.com/@neurocybex/predicting-wine-quality-using-machine-learning-a-step-by-step-guide-aca994bbc128)
- **Dataset**: [Wine Quality Dataset](https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv)
- **GitHub Repository**: [Predicting Wine Quality Based on Chemical Properties](https://github.com/WAJIHAALI1070/100DaysofBytewise/tree/main/Project%20WajihaAli-Week04)




## Dataset
The dataset used is the [Wine Quality Dataset](https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv) sourced from the UCI Machine Learning Repository. It consists of red wine samples with 11 physicochemical properties (features) and a quality rating.

## Requirements
To run this project, ensure you have Python installed along with the following libraries:
- pandas
- matplotlib
- seaborn
- scikit-learn

Install the required libraries using:

    pip install pandas matplotlib seaborn scikit-learn


## Steps Involved
### Loading the Dataset
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns
    from sklearn.model_selection import train_test_split
    from sklearn.linear_model import LinearRegression
    from sklearn.metrics import mean_squared_error, r2_score
    url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv'
    wine_data = pd.read_csv(url, sep=';')
    print(wine_data.describe())

### Data Visualization
Visualize the distribution of wine quality using a count plot and explore relationships between features using a correlation heatmap.

### Distribution of Wine Quality

    sns.countplot(x='quality', data=wine_data)
    plt.title('Wine Quality Distribution')
    plt.show()

### Corelation Heatmap

    plt.figure(figsize=(12, 8))
    sns.heatmap(wine_data.corr(), annot=True, fmt='.2f', cmap='coolwarm')
    plt.title('Correlation Heatmap')    
    plt.show()

### Data Preprocessing
Define the features (input variables) and the target variable (wine quality ratings). Split the dataset into training and testing sets using `train_test_split()`.
    
        X = wine_data.drop('quality', axis=1)
        y = wine_data['quality']
  
    
### Model Building
Initialize a linear regression model using `LinearRegression()` from scikit-learn. Train the model using the training dataset to learn relationships between features and wine quality.
    

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = LinearRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    print(f'Mean Squared Error: {mse}')
    print(f'R-squared: {r2}')


### Model Evaluation
Predict wine quality ratings using the test dataset and evaluate model performance using mean squared error (MSE) and R-squared metrics.

### Results Visualization
Visualize predicted vs actual wine quality using a scatter plot to understand how well the model predicts wine quality.
    
    plt.figure(figsize=(8, 6))
    plt.scatter(y_test, y_pred)
    plt.xlabel('Actual Quality')
    plt.ylabel('Predicted Quality')
    plt.title('Actual vs Predicted Wine Quality')
    plt.tight_layout()
    plt.show()


## Conclusion
The predictive model demonstrates effectiveness in estimating wine quality based on chemical properties. Further improvements could involve exploring advanced machine learning models or incorporating additional features for enhanced predictions.

## Call to Action
Explore this project, try out the code, and delve deeper into machine learning for predictive analytics. Your feedback and comments are highly encouraged to foster learning and community engagement in data science.

