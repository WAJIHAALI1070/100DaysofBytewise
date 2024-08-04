import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
from sklearn.decomposition import PCA
import numpy as np

# Load the Iris dataset
iris = sns.load_dataset('iris')

# Display the first few rows of the dataset
print(iris.head())

# Summary statistics
print(iris.describe())

# Pairplot to visualize relationships between features
sns.pairplot(iris, hue='species')
plt.show()

# Correlation matrix
corr_matrix_iris = iris.corr()
print(corr_matrix_iris)

# Heatmap of the correlation matrix
sns.heatmap(corr_matrix_iris, annot=True, cmap='coolwarm')
plt.title('Correlation Matrix - Iris Dataset')
plt.show()

# Load the Mall Customers dataset
mall_customers = pd.read_csv('Mall_Customers.csv')

# Display the first few rows of the dataset
print(mall_customers.head())

# Summary statistics
print(mall_customers.describe())

# Pairplot to visualize relationships between features (only numerical features)
sns.pairplot(mall_customers[['Age', 'Annual Income (k$)', 'Spending Score (1-100)']])
plt.show()

# Correlation matrix
corr_matrix_mall = mall_customers[['Age', 'Annual Income (k$)', 'Spending Score (1-100)']].corr()
print(corr_matrix_mall)

# Heatmap of the correlation matrix
sns.heatmap(corr_matrix_mall, annot=True, cmap='coolwarm')
plt.title('Correlation Matrix - Mall Customers Dataset')
plt.show()


# Function to determine the optimal number of clusters using the Elbow Method and Silhouette Score
def optimal_clusters(data):
    sse = []
    silhouette_scores = []
    k_range = range(2, 11)

    for k in k_range:
        kmeans = KMeans(n_clusters=k, random_state=42)
        kmeans.fit(data)
        sse.append(kmeans.inertia_)
        silhouette_scores.append(silhouette_score(data, kmeans.labels_))

    return sse, silhouette_scores


# Prepare data for clustering (Iris dataset)
iris_data = iris.drop(columns=['species'])

# Determine optimal clusters for Iris dataset
sse_iris, silhouette_scores_iris = optimal_clusters(iris_data)

# Plot Elbow Method and Silhouette Scores for Iris dataset
plt.figure(figsize=(14, 5))

plt.subplot(1, 2, 1)
plt.plot(range(2, 11), sse_iris, marker='o')
plt.title('Elbow Method - Iris Dataset')
plt.xlabel('Number of Clusters')
plt.ylabel('SSE')

plt.subplot(1, 2, 2)
plt.plot(range(2, 11), silhouette_scores_iris, marker='o')
plt.title('Silhouette Scores - Iris Dataset')
plt.xlabel('Number of Clusters')
plt.ylabel('Silhouette Score')

plt.show()

# Apply K-Means clustering with optimal number of clusters (Iris dataset)
optimal_k_iris = np.argmax(silhouette_scores_iris) + 2
kmeans_iris = KMeans(n_clusters=optimal_k_iris, random_state=42)
iris['kmeans_cluster'] = kmeans_iris.fit_predict(iris_data)

# Prepare data for clustering (Mall Customers dataset)
mall_data = mall_customers[['Age', 'Annual Income (k$)', 'Spending Score (1-100)']]

# Determine optimal clusters for Mall Customers dataset
sse_mall, silhouette_scores_mall = optimal_clusters(mall_data)

# Plot Elbow Method and Silhouette Scores for Mall Customers dataset
plt.figure(figsize=(14, 5))

plt.subplot(1, 2, 1)
plt.plot(range(2, 11), sse_mall, marker='o')
plt.title('Elbow Method - Mall Customers Dataset')
plt.xlabel('Number of Clusters')
plt.ylabel('SSE')

plt.subplot(1, 2, 2)
plt.plot(range(2, 11), silhouette_scores_mall, marker='o')
plt.title('Silhouette Scores - Mall Customers Dataset')
plt.xlabel('Number of Clusters')
plt.ylabel('Silhouette Score')

plt.show()

# Apply K-Means clustering with optimal number of clusters (Mall Customers dataset)
optimal_k_mall = np.argmax(silhouette_scores_mall) + 2
kmeans_mall = KMeans(n_clusters=optimal_k_mall, random_state=42)
mall_customers['kmeans_cluster'] = kmeans_mall.fit_predict(mall_data)


# Function to apply hierarchical clustering and plot dendrogram
def hierarchical_clustering(data, method):
    linked = linkage(data, method=method)

    plt.figure(figsize=(10, 7))
    dendrogram(linked, orientation='top', distance_sort='descending', show_leaf_counts=True)
    plt.title(f'Dendrogram ({method.capitalize()} Linkage) - {data.name} Dataset')
    plt.show()

    return linked


# Apply hierarchical clustering to Iris dataset
iris_data.name = 'Iris'
linked_iris = hierarchical_clustering(iris_data, 'ward')
iris['hierarchical_cluster'] = fcluster(linked_iris, t=optimal_k_iris, criterion='maxclust')

# Apply hierarchical clustering to Mall Customers dataset
mall_data.name = 'Mall Customers'
linked_mall = hierarchical_clustering(mall_data, 'ward')
mall_customers['hierarchical_cluster'] = fcluster(linked_mall, t=optimal_k_mall, criterion='maxclust')


# Function to visualize clusters using PCA
def visualize_clusters_pca(data, labels, title):
    pca = PCA(n_components=2)
    principal_components = pca.fit_transform(data)
    plt.figure(figsize=(8, 6))
    plt.scatter(principal_components[:, 0], principal_components[:, 1], c=labels, cmap='viridis', marker='o')
    plt.title(f'Clusters Visualization with PCA - {title}')
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.colorbar()
    plt.show()


# Visualize K-Means clusters for Iris dataset
visualize_clusters_pca(iris_data, iris['kmeans_cluster'], 'Iris Dataset (K-Means)')

# Visualize Hierarchical clusters for Iris dataset
visualize_clusters_pca(iris_data, iris['hierarchical_cluster'], 'Iris Dataset (Hierarchical)')

# Visualize K-Means clusters for Mall Customers dataset
visualize_clusters_pca(mall_data, mall_customers['kmeans_cluster'], 'Mall Customers Dataset (K-Means)')

# Visualize Hierarchical clusters for Mall Customers dataset
visualize_clusters_pca(mall_data, mall_customers['hierarchical_cluster'], 'Mall Customers Dataset (Hierarchical)')


# Function to analyze clusters
def analyze_clusters(data, original_data, cluster_col):
    return original_data.groupby(cluster_col).mean()


# Analyze K-Means clusters for Iris dataset
print(analyze_clusters(iris_data, iris, 'kmeans_cluster'))

# Analyze Hierarchical clusters for Iris dataset
print(analyze_clusters(iris_data, iris, 'hierarchical_cluster'))

# Analyze K-Means clusters for Mall Customers dataset
print(analyze_clusters(mall_data, mall_customers, 'kmeans_cluster'))

# Analyze Hierarchical clusters for Mall Customers dataset
print(analyze_clusters(mall_data, mall_customers, 'hierarchical_cluster'))


# Function to compare clustering techniques
def compare_clustering(data, kmeans_labels, hierarchical_labels):
    kmeans_silhouette = silhouette_score(data, kmeans_labels)
    hierarchical_silhouette = silhouette_score(data, hierarchical_labels)
    print(f'K-Means Silhouette Score: {kmeans_silhouette}')
    print(f'Hierarchical Clustering Silhouette Score: {hierarchical_silhouette}')
    return kmeans_silhouette, hierarchical_silhouette


# Compare clustering for Iris dataset
compare_clustering(iris_data, iris['kmeans_cluster'], iris['hierarchical_cluster'])

# Compare clustering for Mall Customers dataset
compare_clustering(mall_data, mall_customers['kmeans_cluster'], mall_customers['hierarchical_cluster'])
