#Combining Multiple Datasets in the Movie Lens Dataset
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# File paths
ratings_file = 'ratings.dat'
users_file = 'users.dat'
movies_file = 'movies.dat'

# Define column names for each dataset
ratings_cols = ['UserID', 'MovieID', 'Rating', 'Timestamp']
users_cols = ['UserID', 'Gender', 'Age', 'Occupation', 'Zip-code']
movies_cols = ['MovieID', 'Title', 'Genres']

# Load data into DataFrames
ratings = pd.read_csv(ratings_file, sep='::', header=None, names=ratings_cols, engine='python')
users = pd.read_csv(users_file, sep='::', header=None, names=users_cols, engine='python')
movies = pd.read_csv(movies_file, sep='::', header=None, names=movies_cols, engine='python', encoding='latin-1')

# Merge datasets
merged_data = pd.merge(pd.merge(ratings, users, on='UserID'), movies, on='MovieID')

# Example preprocessing: convert timestamp to datetime
merged_data['Timestamp'] = pd.to_datetime(merged_data['Timestamp'], unit='s')

# Example visualization: ratings distribution
plt.figure(figsize=(10, 6))
sns.histplot(merged_data['Rating'], bins=5, kde=True)
plt.title('Distribution of Ratings')
plt.xlabel('Rating')
plt.ylabel('Count')
plt.show()

# Save the merged dataset to a new CSV file
merged_data.to_csv('combined_movie_lens_data.csv', index=False)

# Display the first few rows of the merged dataset
print(merged_data.head())
