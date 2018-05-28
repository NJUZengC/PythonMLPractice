# -*- coding: utf-8 -*-
# @Time    : 2018/5/14 16:19
# @Author  : ZengC
# @Email   : njuzengc@foxmail.com
# @File    : example1.py
# @Software: PyCharm Community Edition

import tensorflow as tf

matrix1 = tf.constant([[3.,3.]])
matrix2 = tf.constant([[2.],[2.]])

product = tf.matmul(matrix1,matrix2)
linear = tf.add(product,tf.constant(2.0))

with tf.Session() as sess:
    result = sess.run(linear)
    print(result)