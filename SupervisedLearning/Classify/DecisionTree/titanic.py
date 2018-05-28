import pandas as pd
from sklearn.cross_validation import train_test_split
from sklearn.feature_extraction import DictVectorizer
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report
from sklearn.svm import LinearSVC

titanic = pd.read_csv('http://biostat.mc.vanderbilt.edu/wiki/pub/Main/DataSets/titanic.txt')
print(titanic.info())
X = titanic[['pclass','age','sex','name']]
y = titanic['survived']
X['age'].fillna(X['age'].mean(),inplace=True)

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.25,random_state=33)
vec = DictVectorizer(sparse=False)
X_train = vec.fit_transform(X_train.to_dict(orient='record'))
X_test = vec.transform(X_test.to_dict(orient='record'))

dt = DecisionTreeClassifier()
dt.fit(X_train,y_train)
dt_y_predict = dt.predict(X_test)

print('Score DT: ',dt.score(X_test,y_test))
print(classification_report(y_test,dt_y_predict,target_names=['died','survived']))

svm = LinearSVC()
svm.fit(X_train,y_train)
svm_y_predict = svm.predict(X_test)
print('Score SVM : ',svm.score(X_test,y_test))
print(classification_report(y_test,svm_y_predict,target_names=['died','survived']))