#Transforming Variables in the Bike Sharing Dataset
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.preprocessing import PowerTransformer

# Load the datasets
hourly_data = pd.read_csv('hour.csv')
daily_data = pd.read_csv('day.csv')

# Display information about the datasets
print("Hourly Data Information:")
print(hourly_data.info())

print("\nDaily Data Information:")
print(daily_data.info())

# Example of transformations (you can customize as needed)
# For example, applying log transformation to skewed variables
skewed_vars = ['cnt', 'casual', 'registered']  # Variables that may need transformation
hourly_data[skewed_vars] = hourly_data[skewed_vars].apply(lambda x: np.log1p(x))

# Applying Box-Cox transformation to 'temp' variable
pt = PowerTransformer(method='box-cox', standardize=False)
hourly_data['temp_boxcox'] = pt.fit_transform(hourly_data[['temp']])

# Visualize transformations
plt.figure(figsize=(12, 6))
plt.subplot(2, 2, 1)
sns.histplot(hourly_data['cnt'], kde=True)
plt.title('Histogram of cnt (log transformed)')

plt.subplot(2, 2, 2)
sns.histplot(hourly_data['casual'], kde=True)
plt.title('Histogram of casual (log transformed)')

plt.subplot(2, 2, 3)
sns.histplot(hourly_data['registered'], kde=True)
plt.title('Histogram of registered (log transformed)')

plt.subplot(2, 2, 4)
sns.histplot(hourly_data['temp_boxcox'], kde=True)
plt.title('Histogram of temp (Box-Cox transformed)')

plt.tight_layout()
plt.show()
