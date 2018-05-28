# -*- coding: utf-8 -*-
# @Time    : 2018/5/14 13:57
# @Author  : ZengC
# @Email   : njuzengc@foxmail.com
# @File    : newsgroup.py
# @Software: PyCharm Community Edition

from sklearn.datasets import fetch_20newsgroups
from sklearn.cross_validation import train_test_split
import numpy as np

news = fetch_20newsgroups(subset='all')
X,y = news.data,news.target

from bs4 import BeautifulSoup
import nltk,re

def news_to_sentence(news):

    news_text = BeautifulSoup(news,"lxml").get_text()
    tokenlizer = nltk.data.load('tokenizers/punkt/english.pickle')
    raw_sentence = tokenlizer.tokenize(news_text)
    sentence = []

    for sent in raw_sentence:
        sentence.append(re.sub('[^a-zA-Z]',' ',sent.lower().strip()).split())
    return sentence

sentences = []
for x in X:
    sentences += news_to_sentence(x)

from  gensim.models import word2vec
num_features = 300
min_word_count = 20
num_workers = 2

context = 5
downsampling = 1e-3

model = word2vec.Word2Vec(sentences,workers=num_workers,size = num_features,min_count=min_word_count,window=context,sample=downsampling)
model.init_sims(replace=True)
print(model.most_similar('email'))
