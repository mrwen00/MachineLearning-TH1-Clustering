from time import time
import numpy as np
import matplotlib.pyplot as plt

from numpy.random import RandomState
from sklearn import metrics
from sklearn.datasets import fetch_lfw_people
from skimage.feature import local_binary_pattern
from sklearn.cluster import KMeans
from sklearn.datasets import load_digits
from sklearn.decomposition import PCA
from sklearn.preprocessing import scale

import matplotlib.cm as cm


dataset = fetch_lfw_people(min_faces_per_person=70, resize=0.4)
data = dataset.data

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

kmeans = KMeans(init='k-means++', n_clusters=7)
y_pred = kmeans.fit_predict(reduced_data)

plt.subplot(221)
plt.scatter(reduced_data[:, 0], reduced_data[:, 1], c=y_pred)
plt.title("Incorrect Number of Blobs")
plt.show()


print y_pred
