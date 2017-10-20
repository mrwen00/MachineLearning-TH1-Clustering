# 14521097
# Trieu Trang Vinh
# BT3 Chon tap du lieu Face, rut trich dac trung LBP.

from time import time
import numpy as np
import matplotlib.pyplot as plt

from sklearn import metrics, cluster
from sklearn.cluster import KMeans
from sklearn.cluster import DBSCAN
from sklearn.cluster import AgglomerativeClustering
from sklearn.datasets import load_digits
from sklearn.decomposition import PCA
from sklearn.preprocessing import scale

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

# =================================================================

np.random.seed(42)

originalData = PCA(n_components=2).fit_transform(data)

plt.subplot(231)
plt.scatter(originalData[:, 0], originalData[:, 1], c=dataset.target)
plt.title('Bai Tap 3 - True Result')


# #############################################################################
# Visualize the results on PCA-reduced data Kmean

y_pred = KMeans(n_clusters=10).fit_predict(lbp)


plt.subplot(232)
plt.scatter(originalData[:, 0], originalData[:, 1], c=y_pred)
plt.title("Kmean")

print "Kmean percent"
print metrics.adjusted_mutual_info_score(dataset.target, y_pred)  

# =================================================================
# spectral
similar_data = np.corrcoef(lbp)

y = cluster.SpectralClustering(n_clusters=n_clusters, eigen_solver='arpack', affinity='nearest_neighbors').fit_predict(similar_data)

plt.subplot(233)
plt.scatter(originalData[:,0], originalData[:, 1], c=y)
plt.title("Spectral")

print "Spectral percent"
print metrics.adjusted_mutual_info_score(dataset.target, y)  

# =================================================================
# dbscan

dbscan = DBSCAN(eps=17.6, min_samples=1).fit_predict(lbp)
plt.subplot(234)
plt.scatter(originalData[:, 0], originalData[:, 1], c=dbscan)
plt.title('DBSCAN')

print "DBSCAN percent"
print metrics.adjusted_mutual_info_score(dataset.target, dbscan)  

# =================================================================
# agglomerative

agglo = AgglomerativeClustering(n_clusters=n_clusters).fit_predict(lbp)
plt.subplot(235)
plt.scatter(originalData[:, 0], originalData[:, 1], c=agglo)
plt.title('Agglomerative')

print "Agglo percent"
print metrics.adjusted_mutual_info_score(dataset.target, agglo)  

plt.show()
