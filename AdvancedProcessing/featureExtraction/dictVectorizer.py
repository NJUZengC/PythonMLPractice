# -*- coding: utf-8 -*-
# @Time    : 2018/5/11 14:44
# @Author  : ZengC
# @Email   : njuzengc@foxmail.com
# @File    : dictVectorizer.py
# @Software: PyCharm Community Edition
measurements = [{'city':'Dubai','temperature':33.},{'city':'Beijing','temperature':23.},{'city':'Tianjin','temperature':13.}]
from sklearn.feature_extraction import DictVectorizer
vec = DictVectorizer()
print(vec.fit_transform(measurements).toarray())
print(vec.get_feature_names())