# -*- coding: utf-8 -*-
# @Time    : 2018/5/14 15:56
# @Author  : ZengC
# @Email   : njuzengc@foxmail.com
# @File    : titanic.py
# @Software: PyCharm Community Edition
import pandas as pd
from sklearn.cross_validation import train_test_split
from sklearn.feature_extraction import DictVectorizer
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report
from sklearn.svm import LinearSVC

titanic = pd.read_csv('http://biostat.mc.vanderbilt.edu/wiki/pub/Main/DataSets/titanic.txt')
y = titanic['survived']
X = titanic[['pclass','age','sex']]
X['age'].fillna(X['age'].mean(),inplace=True)

from sklearn.cross_validation import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.25,random_state=33)

vec  = DictVectorizer()
X_train = vec.fit_transform(X_train.to_dict(orient='record'))
X_test = vec.transform(X_test.to_dict(orient='record'))

from sklearn.ensemble import RandomForestClassifier
rfc  =RandomForestClassifier()
rfc.fit(X_train,y_train)
print('Score: ',rfc.score(X_test,y_test))

from xgboost import XGBClassifier
xgbc = XGBClassifier()
xgbc.fit(X_train,y_train)
print('Score: ',xgbc.score(X_test,y_test))