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

np.random.seed(42)

digits = load_digits()
data = digits.data

X = PCA(n_components=2).fit_transform(data)

plt.subplot(231)
plt.scatter(X[:, 0], X[:, 1], c=digits.target)
plt.title('Bai Tap 2 - True Result')

n_samples, n_features = data.shape
n_digits = len(np.unique(digits.target))
labels = digits.target


# #############################################################################
# Visualize the results on PCA-reduced data Kmean

reduced_data = PCA(n_components=2).fit_transform(data)
y_pred = KMeans(n_clusters=10).fit_predict(data)


plt.subplot(232)
plt.scatter(reduced_data[:, 0], reduced_data[:, 1], c=y_pred)
plt.title("Kmean")

print "Kmean percent"
print metrics.adjusted_mutual_info_score(digits.target, y_pred)  

# =================================================================
# spectral
similar_data = np.corrcoef(data)

y = cluster.SpectralClustering(n_clusters=10, eigen_solver='arpack', affinity='precomputed').fit_predict(similar_data)
reduced_data = PCA(n_components=2).fit_transform(data)

plt.subplot(233)
plt.scatter(reduced_data[:,0], reduced_data[:, 1], c=y)
plt.title("Spectral")

print "Spectral percent"
print metrics.adjusted_mutual_info_score(digits.target, y)  

# =================================================================
# dbscan

dbscan = DBSCAN(eps=17.6, min_samples=1).fit_predict(data)
plt.subplot(234)
plt.scatter(reduced_data[:, 0], reduced_data[:, 1], c=dbscan)
plt.title('DBSCAN')

print "DBSCAN percent"
print metrics.adjusted_mutual_info_score(digits.target, dbscan)  

# =================================================================
# agglomerative

agglo = AgglomerativeClustering(n_clusters=10).fit_predict(data)
plt.subplot(235)
plt.scatter(reduced_data[:, 0], reduced_data[:, 1], c=agglo)
plt.title('Agglomerative')

print "Agglo percent"
print metrics.adjusted_mutual_info_score(digits.target, agglo)  

plt.show()
