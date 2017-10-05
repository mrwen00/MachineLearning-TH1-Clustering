# 14521097
# Trieu Trang Vinh
# BT1 Thuc hien thuat toan Kmean. Du lieu duoc sinh ngau nghien gom 2 gaussian. Visualize ket qua

import numpy as np
import matplotlib.pyplot as plt
from random import randint

from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs

plt.figure(figsize=(12, 12))

n_samples = 100
random_state = randint(1, 1000)
X, y = make_blobs(n_samples=n_samples,centers = 2, random_state=random_state)

# plt.subplot(221)
# plt.scatter(X[:, 0], X[:, 1], c=y)
# plt.title("Original")

# Incorrect number of clusters
y_pred = KMeans(n_clusters=2, random_state=random_state).fit_predict(X)

plt.subplot(221)
plt.scatter(X[:, 0], X[:, 1], c=y_pred)
plt.title("Kmean")

print X

plt.show()
