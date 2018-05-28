# -*- coding: utf-8 -*-
# @Time    : 2018/5/14 22:58
# @Author  : ZengC
# @Email   : njuzengc@foxmail.com
# @File    : main.py
# @Software: PyCharm Community Edition
import pandas as pd
import os
print(os.getcwd())
train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')
print(train.info())
selected_features = ['Pclass','Sex','Age','Embarked','SibSp','Parch','Fare']
X_train = train[selected_features]
X_test = test[selected_features]

y_train = train['Survived']
X_train['Embarked'].fillna('S',inplace=True)
X_test['Embarked'].fillna('S',inplace=True)

X_train['Age'].fillna(X_train['Age'].mean(),inplace=True)
X_test['Age'].fillna(X_test['Age'].mean(),inplace=True)
X_test['Fare'].fillna(X_train['Fare'].mean(),inplace=True)


from sklearn.feature_extraction import DictVectorizer
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=18)
dict = DictVectorizer()
X_train = dict.fit_transform(X_train.to_dict(orient='record'))
X_test = dict.transform(X_test.to_dict(orient='record'))


from xgboost import XGBClassifier
xgbc = XGBClassifier()
from sklearn.ensemble import GradientBoostingClassifier,ExtraTreesClassifier
from sklearn.tree import DecisionTreeClassifier

gbc = GradientBoostingClassifier(max_leaf_nodes=15)


from sklearn.cross_validation import train_test_split,cross_val_score
from sklearn.svm import LinearSVC
svc = LinearSVC()
print(cross_val_score(xgbc,X_train,y_train,cv=5).mean())
print(cross_val_score(gbc,X_train.toarray(),y_train,cv=5).mean())
print(cross_val_score(knn,X_train.toarray(),y_train,cv=5).mean())
xgbc.fit(X_train,y_train)
gbc.fit(X_train,y_train)
xgbc_y_predict = xgbc.predict(X_test)
xgbc_submission = pd.DataFrame({'PassengerId':test['PassengerId'],'Survived':xgbc_y_predict})
xgbc_submission.to_csv("xgbc_submission.csv",index=False)
gbc_y_predict = gbc.predict(X_test.toarray() )
gbc_submission = pd.DataFrame({'PassengerId':test['PassengerId'],'Survived':gbc_y_predict})
gbc_submission.to_csv("gbc_submission.csv",index=False)
from sklearn.grid_search import GridSearchCV
import numpy as np
print(np.arange(0.05,1.,0.05))
params = {'learning_rate':np.arange(0.05,1.,0.05)}
gbc_c = GradientBoostingClassifier()
gs = GridSearchCV(gbc_c,params,cv=5,verbose=1)
gs.fit(X_train.toarray(),y_train)
print(gs.best_score_)
gbc_y_predict = gs.predict(X_test.toarray() )
gbc_submission = pd.DataFrame({'PassengerId':test['PassengerId'],'Survived':gbc_y_predict})
gbc_submission.to_csv("gbcx_submission.csv",index=False)