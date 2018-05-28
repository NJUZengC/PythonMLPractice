from sklearn.datasets import load_digits
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC,SVC
from sklearn.metrics import classification_report
from sklearn.linear_model import LogisticRegression

digits = load_digits()
print(digits.DESCR)
X_train,X_test,y_train,y_test = train_test_split(digits.data,digits.target,test_size=0.25,random_state=33)

ss = StandardScaler()
X_train = ss.fit_transform(X_train)
X_test = ss.transform(X_test)

svm = LinearSVC()
svm = SVC(kernel='rbf')
svm.fit(X_train,y_train)
svm_y_predict = svm.predict(X_test)
print('Score SVM : ',svm.score(X_test,y_test))
print(classification_report(y_test,svm_y_predict))

lr = LogisticRegression()
lr.fit(X_train,y_train)
lr_y_predict = lr.predict(X_test)
print('Score LR : ',lr.score(X_test,y_test))
print(classification_report(y_test,lr_y_predict))