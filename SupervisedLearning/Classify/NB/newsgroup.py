from sklearn.datasets import fetch_20newsgroups
from sklearn.cross_validation import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import  classification_report
from sklearn.svm import  LinearSVC

news = fetch_20newsgroups()

X_train,X_test,y_train,y_test = train_test_split(news.data,news.target,test_size=0.25,random_state=33)
vec = CountVectorizer()
X_train = vec.fit_transform(X_train)

X_test = vec.transform(X_test)

mnb = MultinomialNB()
mnb.fit(X_train,y_train)
mnb_y_predict = mnb.predict(X_test)

print('Score MNB : ',mnb.score(X_test,y_test))
print(classification_report(y_test,mnb_y_predict))

svm = LinearSVC()
svm.fit(X_train,y_train)
svm_y_predict = svm.predict(X_test)
print('Score SVM : ',svm.score(X_test,y_test))
print(classification_report(y_test,svm_y_predict))