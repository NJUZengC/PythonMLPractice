# -*- coding: utf-8 -*-
# @Time    : 2018/5/14 17:12
# @Author  : ZengC
# @Email   : njuzengc@foxmail.com
# @File    : linearCancer.py
# @Software: PyCharm Community Edition
from sklearn.cross_validation import  train_test_split
import tensorflow as tf
import pandas as pd
import numpy as np

column_names=['Sample code number','Clump Thickness','Uniformity of Cell Size','Uniformity of Cell Shape'
              ,'Marginal Adhesion','Single Epithelial Cell Size','Bare Nuclei','Bland Chromatin','Normal Nucleoli',
              'Mitoses','Class']
data = pd.read_csv('http://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/breast-cancer-wisconsin.data',names=column_names)

data = data.replace(to_replace='?',value=np.nan)
data = data.dropna(how='any')
X_train, X_test, y_train, y_test = train_test_split(data[column_names[0:10]], data[column_names[10]], test_size=0.25,
                                                    random_state=33)
X_train = np.float32(X_train[['Clump Thickness','Uniformity of Cell Size']].T)
y_train = np.float32((y_train.T-2)/2)
print(y_train)
X_test = np.float32(X_test[['Clump Thickness','Uniformity of Cell Size']].T)
y_test = np.float32(y_test.T)

b = tf.Variable(tf.zeros([1]))
W = tf.Variable(tf.random_uniform([1,2],-1.0,1.0))
y = tf.matmul(W,X_train) + b

loss = tf.reduce_mean(tf.square(y-y_train))
optimizer = tf.train.GradientDescentOptimizer(0.01)

train = optimizer.minimize(loss)

init  = tf.initialize_all_variables()

sess = tf.Session()
sess.run(init)

for step in range(0,1000):
    sess.run(train)
    if step%200 == 0:
        print(step,sess.run(W),sess.run(b),sess.run(loss))

X_test = X_test.T
test_negative = []
test_positive = []
for i,j in  enumerate(y_test):
    if j==2:
        test_negative.append(X_test[i])
    else:
        test_positive.append(X_test[i])


import matplotlib.pyplot as plt

test_negative = np.matrix(test_negative)
test_positive = np.matrix(test_positive)
print(test_negative[:,0].reshape(len(test_negative),order='C').tolist()[0])
plt.scatter(test_negative[:,0].reshape(len(test_negative),order='C').tolist()[0],test_negative[:,1].reshape(len(test_negative),order='C').tolist()[0],marker='o',s=200,c='red')
plt.scatter(test_positive[:,0].reshape(len(test_positive),order='C').tolist()[0],test_positive[:,1].reshape(len(test_positive),order='C').tolist()[0],marker='x',s=150,c='black')

lx = np.arange(0,12)

ly = (0.5-sess.run(b)-lx * sess.run(W)[0][0])/sess.run(W)[0][1]
plt.plot(lx,ly,color='green')
print(lx* sess.run(W)[0][0] + ly * sess.run(W)[0][1] + sess.run(b))
plt.show()