"""
Author: Maosen Li
"""
from __future__ import division
import os
import time
import glob
import math

import tensorflow as tf
import numpy as np

from utils import *


def classifier(input_data, hidden_dim, activation_fn, normalizer_fn, scope_name):

	h1 = tf.contrib.layers.fully_connected(input_data, hidden_dim, activation_fn=activation_fn,\
		                                       normalizer_fn=normalizer_fn)
	h2 = tf.contrib.layers.fully_connected(h1, 3, activation_fn=None,\
		                                       normalizer_fn=None)
	out = tf.nn.softmax(h2)

	return h2, out


class EmotionClassifier(object):
	def __init__(self, sess, args):
		self.sess = sess
		self.batch_size = args.batch_size
		self.dataset_dir = args.dataset_dir
		self.checkpoint_dir = args.checkpoint_dir
		self.result_dir = args.result_dir

		self.classifier = classifier
		self.hidden_dim = args.hidden_dim
		self._build_model(args)
		self.saver = tf.train.Saver()


	def _build_model(self, args):
		_, self.data_num, self.data_dim = data_load(os.path.join(self.dataset_dir, args.train_set))
		_, self.label_num, self.label_dim = data_load(os.path.join(self.dataset_dir, args.train_label))

		self.train_ = tf.placeholder(tf.float32,[None, self.data_dim], name='train_data')
		self.train_label = tf.placeholder(tf.int32, [None, 3], name='train_label')
		self.h2, self.predict = self.classifier(self.train_, self.hidden_dim, activation_fn=tf.nn.relu, normalizer_fn=None, scope_name='classifier')

		self.test_ = tf.placeholder(tf.float32,[None, self.data_dim], name='test_data')
		self.test_label = tf.placeholder(tf.int32, [None, 3], name='test_label')
		self.h2_, self.predict_ = self.classifier(self.test_, self.hidden_dim, activation_fn=tf.nn.relu, normalizer_fn=None, scope_name='classifier')

		self.train_loss = tf.reduce_mean(tf.losses.softmax_cross_entropy(self.train_label, self.h2))
		self.test_loss = tf.reduce_mean(tf.losses.softmax_cross_entropy(self.test_label, self.h2_))

		self.train_acc_sum = tf.reduce_sum(tf.cast(tf.equal(tf.argmax(self.predict, axis=1),tf.argmax(self.train_label, axis=1)),tf.int64))
		self.test_acc_sum = tf.reduce_sum(tf.cast(tf.equal(tf.argmax(self.predict_, axis=1),tf.argmax(self.test_label, axis=1)),tf.int64))

		t_vars = tf.trainable_variables()
		self.t_vars = [var for var in t_vars]
		for var in t_vars: print(var.name)


	def train(self, args, input_data, input_label, input_test_data, input_test_label):
		self.l_rate = tf.placeholder(tf.float32, None, name='learning_rate')
		self.optim = tf.train.AdamOptimizer(self.l_rate, beta1=args.beta1).minimize(self.train_loss, var_list=self.t_vars)

		init_op = tf.global_variables_initializer()
		self.sess.run(init_op)

		counter = 1
		start_time = time.time()

		if args.continue_train and self.load(args.checkpoint_dir):
			print(" [*] Load SUCCESS")
		else:
			print(" [!] Load failed...")

		data_label_pair = data_label_combine(input_data, input_label)
		train_num = int(math.ceil(self.data_num/5*4))
		np.random.shuffle(data_label_pair)
		train_pair = data_label_pair[0:train_num]
		valid_pair = data_label_pair[train_num:]
		test_data_label_pair = data_label_combine(input_test_data, input_test_label)

		for epoch in range(args.epoch):
			np.random.shuffle(train_pair)
			batch_idx = train_num//self.batch_size
			lr = args.lr if epoch < args.epoch_step else args.lr*(args.epoch-epoch)/(args.epoch-args.epoch_step)

			for idx in range(batch_idx):
				batch_train_pair = train_pair[idx*self.batch_size:(idx+1)*self.batch_size]
				batch_train_data = []
				batch_train_label = []
				for i in range(self.batch_size):
					single_train_data = batch_train_pair[i][0]
					batch_train_data.append(single_train_data)
					single_train_label = batch_train_pair[i][1]
					batch_train_label.append(single_train_label)
				batch_train_data = np.array(batch_train_data)
				batch_train_label = np.array(batch_train_label)

				training_loss, _, training_acc_sum= self.sess.run([self.train_loss, self.optim, self.train_acc_sum],\
				                                                   feed_dict={self.train_: batch_train_data, self.train_label: batch_train_label, self.l_rate:lr})
				training_acc_rate = training_acc_sum/self.batch_size
				print("Epoch: [%2d] [%5d] loss:[%g] acc:[%g] time: %4.4f" % (epoch, counter, training_loss, training_acc_rate, time.time()-start_time))
				counter += 1

				if np.mod(counter, args.print_freq) == 0:
					valid_acc_rate = self.sample_model(valid_pair)

				if np.mod(counter, args.print_freq) == 0:
					test_acc_rate = self.test_model(test_data_label_pair)
					f1 = open(os.path.join(self.result_dir, ('./test_accuracy_%d.txt')%self.hidden_dim),'a')
					f1.write('%d\t%g\n'%(counter, test_acc_rate))
					f1.close()
					
				if np.mod(counter, args.save_freq) == 0:
					self.save(self.checkpoint_dir, counter)


	def save(self, checkpoint_dir, step):

		model_name = 'emotion_classifier'
		model_dir = '%d_dim_hidden_layer' % self.hidden_dim
		checkpoint_dir = os.path.join(checkpoint_dir, model_dir)

		if not os.path.exists(checkpoint_dir):
			os.makedirs(checkpoint_dir)

		self.saver.save(self.sess, os.path.join(checkpoint_dir, model_name), global_step=step)


	def load(self, checkpoint_dir):

		print(" [*] Reading checkpoint...")

		model_dir = '%d_dim_hidden_layer' % self.hidden_dim
		checkpoint_dir = os.path.join(checkpoint_dir, model_dir)	

		ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
		if ckpt and ckpt.model_checkpoint_path:
			ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
			self.saver.restore(self.sess, os.path.join(checkpoint_dir, ckpt_name))
			print('Load Model from ----------->',checkpoint_dir)
			return True
		else:
			return False


	def sample_model(self, valid_pair):

		batch_valid_data = []
		batch_valid_label = []
		valid_num = len(valid_pair)

		for i in range(valid_num):
			single_valid_data = valid_pair[i][0]
			batch_valid_data.append(single_valid_data)
			single_valid_label = valid_pair[i][1]
			batch_valid_label.append(single_valid_label)

		batch_valid_data = np.array(batch_valid_data)
		batch_valid_label = np.array(batch_valid_label)
		valid_loss, valid_acc_sum = self.sess.run([self.train_loss, self.train_acc_sum],\
						                           feed_dict={self.train_: batch_valid_data, self.train_label: batch_valid_label})

		valid_acc_rate = valid_acc_sum / valid_num
		print('------------>Valid loss: %g, ------------>Valid acc: %g'%(valid_loss, valid_acc_rate))
		return valid_acc_rate


	def test_model(self, test_pair):

		batch_test_data = []
		batch_test_label = []
		test_num = len(test_pair)

		for i in range(test_num):
			single_test_data = test_pair[i][0]
			batch_test_data.append(single_test_data)
			single_test_label = test_pair[i][1]
			batch_test_label.append(single_test_label)

		batch_test_data = np.array(batch_test_data)
		batch_test_label = np.array(batch_test_label)
		test_loss, test_acc_sum = self.sess.run([self.train_loss, self.train_acc_sum],\
						                           feed_dict={self.train_: batch_test_data, self.train_label: batch_test_label})

		test_acc_rate = test_acc_sum / test_num
		print('------------>Test loss: %g, ------------>Test acc: %g'%(test_loss, test_acc_rate))
		return test_acc_rate


	def test(self, args, input_data, input_label):

		init_op = tf.global_variables_initializer()
		self.sess.run(init_op)

		if self.load(args.checkpoint_dir):
			print(" [*] Load SUCCESS, the dir is ------------>", args.checkpoint_dir)
		else:
			print(" [!] Load failed...")

		test_pair = data_label_combine(input_data, input_label)
		test_num = len(test_pair)
		# np.random.shuffle(test_pair)
		test_acc_sum = 0
		for j in range(test_num):
			single_test_data = np.array([test_pair[j][0]])
			single_test_label = np.array([test_pair[j][1]])
			pridect_label, test_acc = self.sess.run([self.predict_, self.test_acc_sum],\
				                     feed_dict={self.test_:single_test_data, self.test_label: single_test_label})
			print(single_test_label, pridect_label)
			test_acc_sum = test_acc_sum + test_acc
		test_acc_rate = test_acc_sum / test_num
		print('------------>Test acc: %g' % test_acc_rate)