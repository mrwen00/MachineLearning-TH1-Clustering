from time import time
import numpy as np
import matplotlib.pyplot as plt

from sklearn import metrics, cluster
from sklearn.cluster import spectral_clustering
from sklearn.datasets import load_digits
from sklearn.decomposition import PCA
from sklearn.preprocessing import scale

np.random.seed(42)

digits = load_digits()
data = digits.data

similar_data = np.corrcoef(data)

y = cluster.SpectralClustering(n_clusters=10, eigen_solver='arpack', affinity='precomputed').fit_predict(similar_data)
reduced_data = PCA(n_components=2).fit_transform(data)
plt.scatter(reduced_data[:,0], reduced_data[:, 1], c=y)
plt.show()
