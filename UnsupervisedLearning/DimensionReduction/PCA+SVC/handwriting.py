# -*- coding: utf-8 -*-
# @Time    : 2018/5/11 14:27
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

from sklearn.svm import LinearSVC
svc = LinearSVC()
svc.fit(X_train,y_train)
svc_y_predict = svc.predict(X_test)


from sklearn.decomposition import PCA
pca = PCA(n_components=30)
X_pca_train = pca.fit_transform(X_train)
X_pca_test = pca.transform(X_test)

pca_svc = LinearSVC()
pca_svc.fit(X_pca_train,y_train)
pca_svc_y_predict = pca_svc.predict(X_pca_test)

from sklearn.metrics import classification_report
print('Score SVC: ',svc.score(X_test,y_test))
print(classification_report(y_test,svc_y_predict))

print('Score PCA+SVC: ',pca_svc.score(X_pca_test,y_test))
print(classification_report(y_test,pca_svc_y_predict))