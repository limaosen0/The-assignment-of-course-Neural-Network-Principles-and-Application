"""
Author: Maosen Li (017034910074), Shanghai Jiao Tong University
"""

import scipy.misc
import os

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np


def get_mnist_data(data_dir, one_hot):
	mnist = input_data.read_data_sets(data_dir, one_hot=one_hot)
	return mnist


def image_save(i, save_dir, image):
	scipy.misc.imsave(os.path.join(save_dir, str(i)+'.png'), image)


def softmax(x, axis=None, name=None, dim=None):
	return tf.nn.softmax(x, axis=axis, name=name, dim=dim)