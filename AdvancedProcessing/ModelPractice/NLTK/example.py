# -*- coding: utf-8 -*-
# @Time    : 2018/5/11 21:20
# @Author  : ZengC
# @Email   : njuzengc@foxmail.com
# @File    : example.py
# @Software: PyCharm Community Edition
sent1 = 'The cat is walking in the bedroom'
sent2 = 'A dog was running across the kitchen'

from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer
vec_count = CountVectorizer()
X_train = [sent1,sent2]
x = vec_count.fit_transform(X_train).toarray()
print(x,vec_count.get_feature_names())

vec_tfidf = TfidfVectorizer()
print(vec_tfidf.fit_transform(X_train).toarray(),vec_tfidf.get_feature_names())

import nltk

tokens_1 = nltk.word_tokenize(sent1)
print(tokens_1)

vocab_1 = sorted(set(tokens_1))
print(vocab_1)

stemmer = nltk.stem.PorterStemmer()
stem_1 = [stemmer.stem(i) for i in tokens_1]
print(stem_1)

post_tag = nltk.tag.pos_tag(tokens_1)
print(post_tag)