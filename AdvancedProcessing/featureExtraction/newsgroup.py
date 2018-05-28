# -*- coding: utf-8 -*-
# @Time    : 2018/5/11 14:49
# @Author  : ZengC
# @Email   : njuzengc@foxmail.com
# @File    : newsgroup.py
# @Software: PyCharm Community Edition
from sklearn.datasets import fetch_20newsgroups
news = fetch_20newsgroups(subset='all')

from sklearn.cross_validation import train_test_split
X_train,X_test,y_train,y_test = train_test_split(news.data,news.target,test_size=0.25,random_state=33)

from sklearn.feature_extraction.text import CountVectorizer
count_vec = CountVectorizer()
X_count_train = count_vec.fit_transform(X_train)
X_count_test = count_vec.transform(X_test)

from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report

mnb = MultinomialNB()
mnb.fit(X_count_train,y_train)
mnb_y_predict = mnb.predict(X_count_test)

print('Score MNB+COUNT without stopword: ',mnb.score(X_count_test,y_test))
#print(classification_report(y_test,mnb_y_predict,target_names=news.target_names))

from sklearn.feature_extraction.text import TfidfVectorizer
tfidf = TfidfVectorizer()
X_tf_train = tfidf.fit_transform(X_train)
X_tf_test = tfidf.transform(X_test)

mnb_tf = MultinomialNB()
mnb_tf.fit(X_tf_train,y_train)
mnb_tf_y_predict = mnb_tf.predict(X_tf_test)

print('Score MNB+TFIDF without stopword: ',mnb_tf.score(X_tf_test,y_test))

count_filter_vec,tfidf_filter_vec = CountVectorizer(analyzer='word',stop_words='english'),TfidfVectorizer(analyzer='word',stop_words='english')
X_count_filter_train = count_filter_vec.fit_transform(X_train)
X_count_filter_test = count_filter_vec.transform(X_test)
X_tfidf_filter_train = tfidf_filter_vec.fit_transform(X_train)
X_tfidf_filter_test = tfidf_filter_vec.transform(X_test)

mnb_filter_count = MultinomialNB()
mnb_filter_count.fit(X_count_filter_train,y_train)
mnb_filter_count_y_predict = mnb_filter_count.predict(X_count_filter_test)
print('Score COUNT STOPWORD: ',mnb_filter_count.score(X_count_filter_test,y_test))

mnb_filter_tfidf = MultinomialNB()
mnb_filter_tfidf.fit(X_tfidf_filter_train,y_train)
mnb_filter_tfidf_y_predict = mnb_filter_tfidf.predict(X_tfidf_filter_test)
print('Score tfidf STOPWORD: ',mnb_filter_tfidf.score(X_tfidf_filter_test,y_test))