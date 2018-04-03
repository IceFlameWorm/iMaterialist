# BATCHING

import numpy as np
import os
import tensorflow as tf
import math

os.environ['CUDA_VISIBLE_DEVICES'] = '2'  # use gpu 1 to train
# BATCH_SIZE = 128
# sample_size = 1000
# epoc_times = 128
# repeat_times = 128
# x = np.random.sample((sample_size, 2))
# dataset = tf.data.Dataset.from_tensor_slices(x).batch(BATCH_SIZE)
#
# dataset = dataset.shuffle(buffer_size=1000, seed=1)
#
# dataset = dataset.repeat(repeat_times)
# iter = dataset.make_one_shot_iterator()
# el = iter.get_next()
# t = el
# print(x)
#
# with tf.Session() as sess:
#     for i in range(epoc_times):
#         iter_times = int(math.ceil(1.0 * sample_size / BATCH_SIZE))
#         for j in range(iter_times):
#             print(sess.run(t))
#             print('iter_times:%d'%j)
#         print('epoc %d' % (i + 1))
# x = np.arange(9.).reshape(3, 3)
# print(x)
# t = np.where(x > 5, x, -1)
# print(t)


# test = tf.Variable(tf.zeros((1, 10), dtype=tf.int64))
# with tf.Session() as sess:
#     sess.run(tf.global_variables_initializer())
#     print(sess.run(test))
#     a = sess.run(test)
#     print(np.shape(test),type(test))
#     print(a,np.shape(a),type(a))

temp = np.random.randint(0,7,(5,5))
print(temp)

a = []
a.append(10)
a = a*10
print(a,len(a))