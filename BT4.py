from time import time
import numpy as np
import matplotlib.pyplot as plt

from numpy.random import RandomState
from sklearn import metrics, cluster
from sklearn.datasets import fetch_lfw_people
from sklearn.cluster import DBSCAN
from sklearn.cluster import spectral_clustering
from sklearn.neighbors import kneighbors_graph
from sklearn.cluster import KMeans
from sklearn.datasets import load_digits
from sklearn.decomposition import PCA
from sklearn.preprocessing import scale


from skimage.feature import hog
from skimage import data, color, exposure, io
from skimage.transform import resize

import matplotlib.cm as cm

from os import listdir
from os.path import isfile, join


listFile = [f for f in listdir('data_train') if isfile(join("data_train", f))]

listFeature = np.array([]).reshape(0,288)
for f in listFile:    
    print f
    img = io.imread('./data_train/' + f)
    img = color.rgb2gray(img)
    img = resize(img, (100,100), mode='reflect')
    # print "this is img shape"
    # print img.shape
    
    hog_image = hog(img, orientations=8, pixels_per_cell=(16, 16),
                    cells_per_block=(1, 1))
    #reduced_data = PCA(n_components=2).fit_transform(hog_image)
    print hog_image.shape
    listFeature = np.append(listFeature, [hog_image], axis=0)

print listFeature

# kmeans = KMeans(init='k-means++', n_clusters=n_clusters)
# y_pred = kmeans.fit_predict(reduced_data)

# plt.subplot(221)
# plt.scatter(reduced_data[:, 0], reduced_data[:, 1], c=y_pred)
# plt.title("Face dataset clustering by Kmean")













