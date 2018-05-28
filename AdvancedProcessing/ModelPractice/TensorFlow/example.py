# -*- coding: utf-8 -*-
# @Time    : 2018/5/14 16:15
# @Author  : ZengC
# @Email   : njuzengc@foxmail.com
# @File    : example.py
# @Software: PyCharm Community Edition
import tensorflow as tf
import numpy as np

greeting = tf.constant('Hello Google Tensorflow')
sess = tf.Session()
result = sess.run(greeting)
print(result)
sess.close()