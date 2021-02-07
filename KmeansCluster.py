#Importing the libraries
import matplotlib
import numpy as np
import matplotlib.pyplot as pl
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Mall_Customers.csv')
X = dataset.iloc[:, [3, 4]].values

# Using the elbow method to find the optimal number of clusters
from sklearn.cluster import KMeans
wcss = []
for i in range(1, 11):
   kmeans = KMeans(n_clusters=i, init='k-means++', max_iter=300, n_init=10, random_state=0)
   kmeans.fit(X)
   wcss.append(kmeans.inertia_)
pl.plot(range(1,11), wcss)
pl.title('The Elbow Method')
pl.xlabel('Number of clusters')
pl.ylabel('WCSS')
pl.show()

# Applying k-means to the Mall dataset
kmeans = KMeans(n_clusters=5, init='k-means++', max_iter=300, n_init=10, random_state=0)
y_kmeans = kmeans.fit_predict(X)

# Visualizing the clusters
pl.style.use('seaborn')
pl.scatter(X[y_kmeans == 0, 0], X[y_kmeans == 0, 1], s=20, c='red', label='Cluster1', marker='x')
pl.scatter(X[y_kmeans == 1, 0], X[y_kmeans == 1, 1], s=20, c='blue', label='Cluster2', marker='v')
pl.scatter(X[y_kmeans == 2, 0], X[y_kmeans == 2, 1], s=20, c='green', label='Cluster3')
pl.scatter(X[y_kmeans == 3, 0], X[y_kmeans == 3, 1], s=20, c='cyan', label='Cluster4')
pl.scatter(X[y_kmeans == 4, 0], X[y_kmeans == 4, 1], s=20, c='magenta', label='Cluster5')
pl.scatter(kmeans.cluster_centers_[:,0], kmeans.cluster_centers_[:, 1], s=80, c='yellow')
pl.legend()
pl.title('Cluster of clients')
pl.xlabel('Annual income (k$)')
pl.ylabel('Spending score (1-100)')
pl.show()


