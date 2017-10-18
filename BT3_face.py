from time import time
import numpy as np
import matplotlib.pyplot as plt

from numpy.random import RandomState
from sklearn import metrics, cluster
from sklearn.datasets import fetch_lfw_people
from sklearn.cluster import DBSCAN
from skimage.feature import local_binary_pattern
from sklearn.cluster import spectral_clustering
from sklearn.neighbors import kneighbors_graph
from sklearn.cluster import KMeans
from sklearn.datasets import load_digits
from sklearn.decomposition import PCA
from sklearn.preprocessing import scale

import matplotlib.cm as cm


dataset = fetch_lfw_people(min_faces_per_person=70, resize=0.4)
data = dataset.data
n_clusters = 7
n_samples, n_features = data.shape

print n_samples, n_features
print dataset.target_names
print len(dataset.data[0])
print len(dataset.data)

radius = 3
n_points = 8 * radius

lbp = local_binary_pattern(data, n_points, radius)

# print len(lbp)
# print len(lbp[0])
# print lbp
# print data

# #############################################################################
# Visualize the results on PCA-reduced data

reduced_data = PCA(n_components=2).fit_transform(lbp)


# =================================================================
# VIsualize Kmean

kmeans = KMeans(init='k-means++', n_clusters=n_clusters)
y_pred = kmeans.fit_predict(reduced_data)

plt.subplot(221)
plt.scatter(reduced_data[:, 0], reduced_data[:, 1], c=y_pred)
plt.title("Face dataset clustering by Kmean")

# =================================================================
# spectral clustering

similar_data = np.corrcoef(lbp)
print similar_data

y = cluster.SpectralClustering(
        n_clusters=n_clusters, eigen_solver='arpack',
        affinity="nearest_neighbors").fit_predict(similar_data)
plt.subplot(222)
plt.scatter(reduced_data[:, 0], reduced_data[:, 1], c=y)
plt.title("Face dataset clustering by Spectral")


# =================================================================
# aggro
similar_data = np.corrcoef(lbp)

connectivity = kneighbors_graph(
    lbp, n_neighbors=10, include_self=False)
# make connectivity symmetric
connectivity = 0.5 * (connectivity + connectivity.T)

y = cluster.AgglomerativeClustering(
        linkage="average", affinity="cityblock",
        n_clusters=n_clusters, connectivity=connectivity).fit_predict(similar_data)
plt.subplot(223)
plt.scatter(reduced_data[:, 0], reduced_data[:, 1], c=y)
plt.title("Face dataset clustering by Aggro")

# =================================================================
# dbscan

similar_data = np.corrcoef(lbp)
print similar_data

y = cluster.DBSCAN(eps=0.355).fit_predict(similar_data)
plt.subplot(224)
plt.scatter(reduced_data[:, 0], reduced_data[:, 1], c=y)
plt.title("Face dataset clustering by DBSCAN")

plt.show()
