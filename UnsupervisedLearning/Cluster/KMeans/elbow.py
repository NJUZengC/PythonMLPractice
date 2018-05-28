# -*- coding: utf-8 -*-
# @Time    : 2018/5/10 20:56
# @Author  : ZengC
# @Email   : njuzengc@foxmail.com
# @File    : elbow.py
# @Software: PyCharm Community Edition
import numpy as np
from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist
import matplotlib.pyplot as plt

cluster1 = np.random.uniform(0.5,2.0,(2,10))
cluster2 = np.random.uniform(4.0,6.0,(2,10))
cluster3 = np.random.uniform(8.0,9.5,cdist(X,kmeans.cluster_centers_,'euclidean')(2,10))

X = np.hstack((cluster1,cluster2,cluster3)).T
plt.scatter(X[:,0],X[:,1])
plt.xlabel('x1')
plt.xlabel('x2')
plt.show()

K = range(1,10)
meandistortions = []
for k in K:
    kmeans = KMeans(n_clusters=k)
    kmeans.fit(X)
    meandistortions.append(sum(np.min(cdist(X,kmeans.cluster_centers_,'euclidean'),axis=1))/X.shape[0])
plt.plot(K,meandistortions,'bx-')
plt.xlabel('k')
plt.ylabel('Average Dispersion')
plt.title('Selecting k with the Elbow Method')
plt.show()