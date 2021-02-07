#Importing the libraries
import matplotlib
import numpy as np
import matplotlib.pyplot as pl
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Mall_Customers.csv')
X = dataset.iloc[:, [3, 4]].values

# Using the dendeogram to find the optimal number of clusters
import scipy.cluster.hierarchy as sch
dendrogram = sch.dendrogram(sch.linkage(X, method='ward'))
pl.title('Dendrogram')
pl.xlabel('Customers')
pl.ylabel('Euclidean distances')
pl.show()

# Fitting hierarchical clustering to the dataset
from sklearn.cluster import AgglomerativeClustering
hc = AgglomerativeClustering(n_clusters=5, affinity='euclidean', linkage='ward')
y_hc = hc.fit_predict(X)

# Visualizing the clusters
pl.style.use('seaborn')
pl.scatter(X[y_hc == 0, 0], X[y_hc == 0, 1], s=20, c='red', label='Cluster1', marker='x')
pl.scatter(X[y_hc == 1, 0], X[y_hc == 1, 1], s=20, c='blue', label='Cluster2', marker='v')
pl.scatter(X[y_hc == 2, 0], X[y_hc == 2, 1], s=20, c='green', label='Cluster3')
pl.scatter(X[y_hc == 3, 0], X[y_hc == 3, 1], s=20, c='cyan', label='Cluster4')
pl.scatter(X[y_hc == 4, 0], X[y_hc == 4, 1], s=20, c='magenta', label='Cluster5')
pl.legend()
pl.title('Cluster of clients')
pl.xlabel('Annual income (k$)')
pl.ylabel('Spending score (1-100)')
pl.show()
