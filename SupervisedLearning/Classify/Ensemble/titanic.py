import pandas as pd
from sklearn.cross_validation import train_test_split
from sklearn.feature_extraction import DictVectorizer
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report
from sklearn.svm import LinearSVC

titanic = pd.read_csv('http://biostat.mc.vanderbilt.edu/wiki/pub/Main/DataSets/titanic.txt')
X = titanic[['pclass','age','sex']]
y = titanic['survived']
X['age'].fillna(X['age'].mean(),inplace=True)

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.25,random_state=33)
vec = DictVectorizer(sparse=False)
X_train = vec.fit_transform(X_train.to_dict(orient='record'))
X_test = vec.transform(X_test.to_dict(orient='record'))

dt = DecisionTreeClassifier()
dt.fit_transform(X_train,y_train)
dt_y_predict = dt.predict(X_test)

from sklearn.ensemble import RandomForestClassifier
rfc = RandomForestClassifier()
rfc.fit(X_train,y_train)
rfc_y_predict = rfc.predict(X_test)

from sklearn.ensemble import GradientBoostingClassifier
gbc = GradientBoostingClassifier()
gbc.fit(X_train,y_train)
gbc_y_predict = gbc.predict(X_test)

from sklearn.metrics import classification_report
print('Score DT: ',dt.score(X_test,y_test))
print(classification_report(y_test,dt_y_predict))

print('Score RFC: ',rfc.score(X_test,y_test))
print(classification_report(y_test,rfc_y_predict))

print('Score GBC: ',gbc.score(X_test,y_test))
print(classification_report(y_test,gbc_y_predict))