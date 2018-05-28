# -*- coding: utf-8 -*-
# @Time    : 2018/5/14 22:25
# @Author  : ZengC
# @Email   : njuzengc@foxmail.com
# @File    : boston.py
# @Software: PyCharm Community Edition
from sklearn import datasets,metrics,preprocessing,cross_validation

boston = datasets.load_boston()
X,y = boston.data,boston.target

X_train,X_test,y_train,y_test = cross_validation.train_test_split(X,y,test_size=0.25,random_state=33)

ss = preprocessing.StandardScaler()
X_train = ss.fit_transform(X_train)
X_test = ss.transform(X_test)

import skflow
tf_lr = skflow.TensorFlowLinearRegressor(steps=10000,learning_rate=0.01,batch_size=50)
tf_lr.fit(X_train,y_train)
tf_lr_y_predict = tf_lr.predict(X_test)

print('Score: ',metrics.mean_squared_error(tf_lr_y_predict,y_test))
print('Score: ',metrics.mean_absolute_error(tf_lr_y_predict,y_test))
print('Score: ',metrics.r2_score(tf_lr_y_predict,y_test))