import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans

# Load customer data
customer_data = pd.read_csv('dataset.csv')
print("Dataset Description\n",customer_data.describe())

print("\n\n null values \n")
print(customer_data.isnull().sum())
# Selecting the columns for clustering
X = customer_data.iloc[:, [1, 3]].values  # column 1=products purchased, column 3= money spent

# KMeans clustering and WCSS (Within-Cluster Sum of Squares)
wcss = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, init='k-means++', random_state=42)
    kmeans.fit(X)
    wcss.append(kmeans.inertia_)

# Plot the elbow graph
sns.set()
plt.plot(range(1, 11), wcss)
plt.title('The Elbow Point Graph')
plt.xlabel('Number of Clusters')
plt.ylabel('WCSS')
plt.show()

# Applying KMeans clustering with 5 clusters
kmeans = KMeans(n_clusters=5, init='k-means++', random_state=0)
Y = kmeans.fit_predict(X)

# Adding cluster labels to the original dataset
customer_data['Cluster'] = Y

sns.pairplot(customer_data, hue='Cluster', palette='bright', vars=['product purchased', 'complaints', 'money spent'])
plt.show()

# Scatter plot with clusters
plt.figure(figsize=(7, 5))  # Adjust the figure size if needed
plt.scatter(X[Y == 0, 0], X[Y == 0, 1], s=20, c='green', label='Cluster 1')
plt.scatter(X[Y == 1, 0], X[Y == 1, 1], s=20, c='red', label='Cluster 2')
plt.scatter(X[Y == 2, 0], X[Y == 2, 1], s=20, c='orange', label='Cluster 3')
plt.scatter(X[Y == 3, 0], X[Y == 3, 1], s=20, c='violet', label='Cluster 4')
plt.scatter(X[Y == 4, 0], X[Y == 4, 1], s=20, c='blue', label='Cluster 5')

# Plot centroids
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s=50, c='black', label='Centroids')

plt.title('Customer Groups')
plt.xlabel('Product purchased')
plt.ylabel('Money spent')

# Place the legend outside the plot
plt.legend(loc='center left', bbox_to_anchor=(1, 0.5), title="Clusters")

plt.tight_layout()  # Adjust layout to prevent overlap
plt.show()
