# -*- coding: utf-8 -*-
# @Time    : 2018/5/11 14:06
# @Author  : ZengC
# @Email   : njuzengc@foxmail.com
# @File    : handwriting.py
# @Software: PyCharm Community Edition

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

digits_train = pd.read_csv('http://archive.ics.uci.edu/ml/machine-learning-databases/optdigits/optdigits.tra',header=None)
digits_test = pd.read_csv('http://archive.ics.uci.edu/ml/machine-learning-databases/optdigits/optdigits.tes',header=None)

X_train = digits_train[np.arange(64)]
y_train = digits_train[64]

X_test = digits_test[np.arange(64)]
y_test = digits_test[64]

from sklearn.decomposition import PCA
estimator = PCA(n_components=2)
X_pca = estimator.fit_transform(X_train)


import matplotlib.pyplot as plt
colors = ['black','blue','purple','yellow','white','red','lime','cyan','orange','gray']
for i in range(len(colors)):
    px = X_pca[:,0][y_train.as_matrix() == i]
    py = X_pca[:,1][y_train.as_matrix() == i]
    plt.scatter(px,py,c=colors[i])
plt.legend(np.arange(0,12).astype(str))
plt.xlabel('First Principal Component')
plt.ylabel('Second Principal Component')
plt.show()