# Wholesale Customers Segmentation

This repository contains the implementation of customer segmentation using the Wholesale Customers dataset. The tasks include applying K-Means clustering and hierarchical clustering to segment customers based on their annual spending in different categories. The optimal number of clusters is determined using the Elbow Method and Silhouette Score. Additionally, the effectiveness of K-Means and hierarchical clustering is compared.

## Dataset

The dataset used is the Wholesale Customers dataset. It contains the annual spending of customers in various categories.

## Tasks

### 1. K-Means Clustering for Customer Segmentation

- **Objective**: Use K-Means clustering to segment customers based on their annual spending.
- **Visualization**: Scatter plots are used to visualize the clusters formed.

### 2. Evaluating the Optimal Number of Clusters

- **Objective**: Determine the optimal number of clusters using the Elbow Method and Silhouette Score.
- **Visualization**: Results are visualized to justify the choice of the optimal number of clusters.

### 3. Cluster Analysis and Interpretation

- **Objective**: Interpret the clusters formed in the dataset.
- **Analysis**: Identify the characteristics and differences among the clusters based on spending behavior.

### 4. Hierarchical Clustering: Dendrogram and Cluster Formation

- **Objective**: Apply hierarchical clustering and visualize the dendrogram.
- **Comparison**: Compare the cluster assignments with those obtained from K-Means clustering.

### 5. Comparison of Clustering Results

- **Objective**: Compare the effectiveness of K-Means and hierarchical clustering.
- **Discussion**: Discuss the results in terms of cluster cohesion and separation.

## Installation

To run the code, you need to install the required libraries. You can install them using the following command:

```python
pip install pandas numpy matplotlib seaborn scikit-learn scipy
```
### Visualization
The repository includes visualizations for:

- K-Means Clustering
- Elbow Method and Silhouette Score
- Hierarchical Clustering Dendrogram
- Comparison of K-Means and Hierarchical Clustering
### Files
- `customer_segmentation.py`: Python script for performing customer segmentation.
- `customer_segmentation.ipynb`: Jupyter notebook for performing customer segmentation.
- `Wholesale customers data.csv`: The Wholesale Customers dataset (you need to place this file in the repository).
### Results
The results of the clustering algorithms, along with the visualizations, are included in the repository. The comparisons between K-Means and hierarchical clustering are discussed in terms of cluster cohesion and separation.
### License
This project is licensed under the MIT License. See the LICENSE file for more details.

### Contact
For any queries or issues, please contact your email.
