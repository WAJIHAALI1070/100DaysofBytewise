# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
from sklearn.decomposition import PCA

# Load the dataset from an Excel file
data = pd.read_excel('Mall Customers.xlsx')

# Select the features for clustering
X = data[['Annual Income (k$)', 'Spending Score (1-100)']]

# Apply K-Means clustering
kmeans = KMeans(n_clusters=5, random_state=42)
data['Cluster'] = kmeans.fit_predict(X)

# Create a figure and axis objects
fig, axs = plt.subplots(3, 2, figsize=(18, 18))
fig.suptitle('Customer Segmentation Analysis')

# Plot K-Means clusters
sns.scatterplot(x='Annual Income (k$)', y='Spending Score (1-100)', hue='Cluster', data=data, palette='viridis',
                ax=axs[0, 0])
axs[0, 0].set_title('K-Means Clustering')
axs[0, 0].legend(loc='upper right')

# Elbow Method
sse = []
for k in range(1, 11):
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(X)
    sse.append(kmeans.inertia_)

axs[0, 1].plot(range(1, 11), sse, marker='o')
axs[0, 1].set_title('Elbow Method')
axs[0, 1].set_xlabel('Number of clusters')
axs[0, 1].set_ylabel('SSE')

# Silhouette Score
silhouette_scores = []
for k in range(2, 11):
    kmeans = KMeans(n_clusters=k, random_state=42)
    preds = kmeans.fit_predict(X)
    score = silhouette_score(X, preds)
    silhouette_scores.append(score)

axs[1, 0].plot(range(2, 11), silhouette_scores, marker='o')
axs[1, 0].set_title('Silhouette Score')
axs[1, 0].set_xlabel('Number of clusters')
axs[1, 0].set_ylabel('Silhouette Score')

# Hierarchical Clustering Dendrogram
Z = linkage(X, method='ward')
dendrogram(Z, ax=axs[1, 1])
axs[1, 1].set_title('Hierarchical Clustering Dendrogram')
axs[1, 1].set_xlabel('Customers')
axs[1, 1].set_ylabel('Distance')

# Apply Hierarchical Clustering
data['HCluster'] = fcluster(Z, t=5, criterion='maxclust')

# Plot Hierarchical Clustering
sns.scatterplot(x='Annual Income (k$)', y='Spending Score (1-100)', hue='HCluster', data=data, palette='viridis',
                ax=axs[2, 0])
axs[2, 0].set_title('Hierarchical Clustering')
axs[2, 0].legend(loc='upper right')

# Apply PCA
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)
data['PCA1'] = X_pca[:, 0]
data['PCA2'] = X_pca[:, 1]

# Plot K-Means Clusters in PCA-reduced space
sns.scatterplot(x='PCA1', y='PCA2', hue='Cluster', data=data, palette='viridis', ax=axs[2, 1])
axs[2, 1].set_title('K-Means Clusters in PCA-reduced Space')
axs[2, 1].legend(loc='upper right')

# Plot Hierarchical Clusters in PCA-reduced space
sns.scatterplot(x='PCA1', y='PCA2', hue='HCluster', data=data, palette='viridis', ax=axs[2, 1])
axs[2, 1].set_title('Hierarchical Clusters in PCA-reduced Space')
axs[2, 1].legend(loc='upper right')

# Adjust layout
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show()
