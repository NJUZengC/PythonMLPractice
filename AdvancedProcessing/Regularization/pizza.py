# -*- coding: utf-8 -*-
# @Time    : 2018/5/11 16:01
# @Author  : ZengC
# @Email   : njuzengc@foxmail.com
# @File    : pizza.py
# @Software: PyCharm Community Edition
X_train = [[6],[8],[10],[14],[18]]
y_train = [[7],[9],[13],[17.5],[18]]

from sklearn.linear_model import LinearRegression
lr = LinearRegression()
lr.fit(X_train,y_train)

import numpy as np
xx = np.linspace(0,26,100)
xx = xx.reshape(xx.shape[0],1)
yy = lr.predict(xx)

import matplotlib.pyplot as plt
plt.scatter(X_train,y_train)

plt1, = plt.plot(xx,yy,label='Degree=1')
plt.axis([0,25,0,25])
plt.xlabel('Diameter of Pizza')
plt.ylabel('Price of Pizza')
plt.legend(handles = [plt1])
# plt.show()
print('Score : ',lr.score(X_train,y_train))

from sklearn.preprocessing import PolynomialFeatures
poly2 = PolynomialFeatures(degree=2)
X_train_poly2 = poly2.fit_transform(X_train)

lr_poly2 = LinearRegression()
lr_poly2.fit(X_train_poly2,y_train)

xx_poly2 = poly2.transform(xx)
yy_poly2 = lr_poly2.predict(xx_poly2)
plt.scatter(X_train,y_train)
plt1, = plt.plot(xx,yy,label='Degree=1')
plt2, = plt.plot(xx,yy_poly2,label='Degree=2')
plt.axis([0,25,0,25])
plt.xlabel('Diameter of Pizza')
plt.ylabel('Price of Pizza')
plt.legend(handles = [plt1,plt2])
# plt.show()
print('Score : ',lr_poly2.score(X_train_poly2,y_train))

poly4 = PolynomialFeatures(degree=4)
X_train_poly4 = poly4.fit_transform(X_train)
lr_poly4 = LinearRegression()
lr_poly4.fit(X_train_poly4,y_train)
xx_poly4 = poly4.transform(xx)
yy_poly4 = lr_poly4.predict(xx_poly4)
print('Score: ',lr_poly4.score(X_train_poly4,y_train))

plt.scatter(X_train,y_train)
plt1, = plt.plot(xx,yy,label='Degree=1')
plt2, = plt.plot(xx,yy_poly2,label='Degree=2')
plt3, = plt.plot(xx,yy_poly4,label='Degree=4')
plt.axis([0,25,0,25])
plt.xlabel('Diameter of Pizza')
plt.ylabel('Price of Pizza')
plt.legend(handles = [plt1,plt2,plt3])
# plt.show()

X_test = [[6],[8],[11],[16]]
y_test = [[8],[12],[15],[18]]
print('Score: ',lr.score(X_test,y_test))

X_test_poly2 = poly2.transform(X_test)
print('Score: ',lr_poly2.score(X_test_poly2,y_test))

print('Score: ',lr_poly4.score(poly4.transform(X_test),y_test))

from sklearn.linear_model import Lasso
lasso_poly4 = Lasso()
lasso_poly4.fit(X_train_poly4,y_train)
print('Score : Lasso ',lasso_poly4.score(poly4.transform(X_test),y_test))
print(lasso_poly4.coef_)

from sklearn.linear_model import Ridge
ridge_poly4 = Ridge()
ridge_poly4.fit(X_train_poly4,y_train)
print('Score Ridge: ',ridge_poly4.score(poly4.transform(X_test),y_test))
print(ridge_poly4.coef_)