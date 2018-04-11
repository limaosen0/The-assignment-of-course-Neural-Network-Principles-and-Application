"""
Author: Maosen Li (017034910074), Shanghai Jiao Tong University
"""

import tensorflow as tf
import numpy as np
from utils import *


def nn_classifier(input_data, layers_num, hidden_dim, activation_fn, normalizer_fn, scope_name, dropout, keep_prob):

	h = tf.contrib.layers.fully_connected(input_data, hidden_dim, activation_fn=activation_fn,\
		                                       normalizer_fn=normalizer_fn, scope='0')
	if dropout:
		h = tf.nn.dropout(h, keep_prob)

	for i in range(1, layers_num):
		h = tf.contrib.layers.fully_connected(h, hidden_dim, activation_fn=activation_fn,\
		                                       normalizer_fn=normalizer_fn, scope=str(i))
		if dropout:
			h = tf.nn.dropout(h, keep_prob)

	h = tf.contrib.layers.fully_connected(h, 10, activation_fn=None,\
		                                       normalizer_fn=None)
	out = softmax(h)

	return h, out


def lenet_5(input_data, activation_fn, normalizer_fn):
	
	h1 = tf.contrib.layers.conv2d(input_data, num_outputs=6, kernel_size=5, stride=1,\
		                          padding='VALID', activation_fn=activation_fn,\
		                          weights_initializer=tf.random_normal_initializer(0, 0.02),\
		                          normalizer_fn=normalizer_fn)

	h2 = tf.contrib.layers.max_pool2d(h1, kernel_size=2, stride=2, padding='VALID')

	h3 = tf.contrib.layers.conv2d(h2, num_outputs=16, kernel_size=5, stride=1,\
	                              padding='VALID', activation_fn=activation_fn,\
	                              weights_initializer=tf.random_normal_initializer(0, 0.02),\
	                              normalizer_fn=normalizer_fn)

	h4 = tf.contrib.layers.max_pool2d(h3, kernel_size=2, stride=2, padding='VALID')
	h4 = tf.contrib.slim.flatten(h4)

	h5 = tf.contrib.layers.fully_connected(h4, 120, activation_fn=activation_fn,\
		                                   normalizer_fn=normalizer_fn)

	h6 = tf.contrib.layers.fully_connected(h5, 84, activation_fn=activation_fn,\
		                                   normalizer_fn=normalizer_fn)

	h7 = tf.contrib.layers.fully_connected(h6, 10, activation_fn=None, normalizer_fn=None)

	out = softmax(h7)

	return h1, h3, h7, out