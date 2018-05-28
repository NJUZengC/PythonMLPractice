from sklearn.datasets import load_iris
from sklearn.cross_validation import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report
from sklearn.svm import  LinearSVC

iris = load_iris()
# print(iris.DESCR)

X_train,X_test,y_train,y_test = train_test_split(iris.data,iris.target,test_size=0.25,random_state=33)
ss = StandardScaler()
X_train = ss.fit_transform(X_train)
X_test = ss.transform(X_test)

knn = KNeighborsClassifier()
knn.fit(X_train,y_train)
knn_y_predict = knn.predict(X_test)

print('Score KNN : ',knn.score(X_test,y_test))
print(classification_report(y_test,knn_y_predict,target_names=iris.target_names))

svm = LinearSVC()
svm.fit(X_train,y_train)
svm_y_predict = svm.predict(X_test)
print('Score SVM : ',svm.score(X_test,y_test))
print(classification_report(y_test,svm_y_predict))