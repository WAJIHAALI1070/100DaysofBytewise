#Task18
# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster

# Load the dataset from the specified path
file_path = r'C:\Users\pc\PycharmProjects\100DaysOfBytewise_Task18\Wholesale customers data.csv'
data = pd.read_csv(file_path)

# Exploratory Data Analysis (EDA)
print(data.info())
print(data.describe())
sns.pairplot(data)
plt.show()

# K-Means Clustering
X = data.drop(['Channel', 'Region'], axis=1)
kmeans = KMeans(n_clusters=3, random_state=42)
clusters = kmeans.fit_predict(X)
data['KMeans_Cluster'] = clusters

# Visualize the clusters
plt.figure(figsize=(10, 6))
sns.scatterplot(data=data, x='Fresh', y='Milk', hue='KMeans_Cluster', palette='viridis')
plt.title('K-Means Clustering')
plt.show()

# Elbow Method to determine the optimal number of clusters
sse = []
for k in range(1, 11):
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(X)
    sse.append(kmeans.inertia_)

plt.figure(figsize=(10, 6))
plt.plot(range(1, 11), sse, marker='o')
plt.title('Elbow Method')
plt.xlabel('Number of clusters')
plt.ylabel('SSE')
plt.show()

# Silhouette Score to determine the optimal number of clusters
silhouette_scores = []
for k in range(2, 11):
    kmeans = KMeans(n_clusters=k, random_state=42)
    labels = kmeans.fit_predict(X)
    silhouette_scores.append(silhouette_score(X, labels))

plt.figure(figsize=(10, 6))
plt.plot(range(2, 11), silhouette_scores, marker='o')
plt.title('Silhouette Score Method')
plt.xlabel('Number of clusters')
plt.ylabel('Silhouette Score')
plt.show()

# Hierarchical Clustering
linked = linkage(X, method='ward')
plt.figure(figsize=(10, 7))
dendrogram(linked, orientation='top', distance_sort='descending', show_leaf_counts=True)
plt.title('Hierarchical Clustering Dendrogram')
plt.show()

# Comparing cluster assignments
hc_clusters = fcluster(linked, 3, criterion='maxclust')
data['Hierarchical_Cluster'] = hc_clusters

# Compare K-Means and Hierarchical Clustering
plt.figure(figsize=(10, 6))
sns.scatterplot(data=data, x='Fresh', y='Milk', hue='Hierarchical_Cluster', palette='viridis')
plt.title('Hierarchical Clustering')
plt.show()

plt.figure(figsize=(10, 6))
sns.scatterplot(data=data, x='Fresh', y='Milk', hue='KMeans_Cluster', style='Hierarchical_Cluster', palette='viridis')
plt.title('Comparison of K-Means and Hierarchical Clustering')
plt.show()
