"""
Author: Maosen Li (017034910074), Shanghai Jiao Tong University
"""

from __future__ import division
import os
import time

import utils
from module import nn_classifier, lenet_5

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np


class NN_Classifier(object):

	def __init__(self, sess, args):
		super(NN_Classifier, self).__init__()

		self.dataset_dir = args.dataset_dir
		self.checkpoint_dir = args.checkpoint_dir
		self.result_dir = args.result_dir
		self.nn_classifier = nn_classifier

		self.sess = sess
		self.lr = args.lr
		self.hidden_dim = args.hidden_dim
		self.layers_num = args.layers_num
		self.dropout = args.dropout
		self.keep_prob = args.keep_prob
		self.model_name = 'fc_classifier_' + str(self.layers_num) + '_' + str(self.hidden_dim)

		if args.data_name == 'mnist':
			self.image_size = 28
			self.c_dim = 1
			self.mnist = utils.get_mnist_data(os.path.join(self.dataset_dir, args.data_name), one_hot=True)
			self.batch_size = args.batch_size
			self.vali_batch_size = args.vali_batch_size

		self._build_model()
		self.saver = tf.train.Saver()


	def _build_model(self):

		self.input_image = tf.placeholder(tf.float32, [None, (self.image_size ** 2) * self.c_dim], name='input_image')
		self.real_label = tf.placeholder(tf.float32, [None, 10], name='real_label')

		self.logits, self.pred_label = self.nn_classifier(self.input_image, self.hidden_dim,\
		                                                  tf.nn.relu, None, 'classifier', self.dropout, self.keep_prob)

		self.loss = tf.reduce_mean(tf.losses.softmax_cross_entropy(self.real_label, self.logits))
		self.train_op = tf.train.AdamOptimizer(self.lr).minimize(self.loss)

		self.train_acc_sum = tf.reduce_sum(tf.cast(tf.equal(tf.argmax(self.pred_label, axis=1),tf.argmax(self.real_label, axis=1)),tf.int64))

		t_vars = tf.trainable_variables()
		self.t_vars = [var for var in t_vars]
		for var in t_vars: print(var.name)


	def train(self, args):

		init_op = tf.global_variables_initializer()
		self.sess.run(init_op)

		start_time = time.time()

		for i in range(args.max_iteration):
			self.x_trn, self.y_trn = self.mnist.train.next_batch(self.batch_size)
			self.x_trn = np.reshape(self.x_trn, (self.batch_size, (self.image_size ** 2) * self.c_dim))
			loss, _ = self.sess.run([self.loss, self.train_op], \
				                     feed_dict={self.input_image: self.x_trn, self.real_label: self.y_trn})
			classify_sum = self.sess.run(self.train_acc_sum, \
				                     feed_dict={self.input_image: self.x_trn, self.real_label: self.y_trn})
			classify_acc = classify_sum/self.batch_size
			print("Iter: [%5d] loss:[%4.4f] acc:[%g] time: %4.4f" % (i, loss, classify_acc, time.time()-start_time))

			if np.mod(i, args.print_freq) == 0:
				valid_acc_rate, valid_loss = self.sample()
				print('~~~~~~~~~~~~~~Valid loss is %4.4f, and classify acc is %4.4f'%(valid_loss, valid_acc_rate))

				if not os.path.exists(os.path.join(self.result_dir, self.model_name)):
					os.makedirs(os.path.join(self.result_dir, self.model_name))

				with open(os.path.join(self.result_dir, self.model_name,\
				          'training_loss_'+self.model_name+'.txt'), 'a') as f1:
					f1.write('%d \t %4.4f \n' % (i, loss))

				with open(os.path.join(self.result_dir, self.model_name,\
					      'training_acc_'+self.model_name+'.txt'), 'a') as f2:
					f2.write('%d \t %4.4f \n' % (i, classify_acc))

				with open(os.path.join(self.result_dir, self.model_name,\
					      'validation_loss_'+self.model_name+'.txt'), 'a') as f3:
					f3.write('%d \t %4.4f \n' % (i,valid_loss))

				with open(os.path.join(self.result_dir, self.model_name,\
					      'validation_acc_'+self.model_name+'.txt'), 'a') as f4:
					f4.write('%d \t %4.4f \n' % (i,valid_acc_rate))

			if np.mod(i, args.save_freq) == 0:
				self.save(self.checkpoint_dir, self.model_name, i)
			

	def sample(self):

		self.x_tst, self.y_tst = self.mnist.test.next_batch(self.vali_batch_size)
		valid_loss, valid_acc_sum = self.sess.run([self.loss, self.train_acc_sum],\
						                           feed_dict={self.input_image: self.x_tst, self.real_label: self.y_tst})
		valid_acc_rate = valid_acc_sum / self.vali_batch_size

		return valid_acc_rate, valid_loss


	def load(self, checkpoint_dir, model_name):

		checkpoint_dir = os.path.join(checkpoint_dir, model_name)

		print('****************Loading the Saved Model from****************')
		print(checkpoint_dir)

		ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
		if ckpt and ckpt.model_checkpoint_path:
			ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
			self.saver.restore(self.sess, os.path.join(checkpoint_dir, ckpt_name))
			return True
		else:
			return False


	def test(self, args):

		init_op = tf.global_variables_initializer()
		self.sess.run(init_op)

		if self.load(args.checkpoint_dir, self.model_name):
			print(" [*] Load SUCCESS, the dir is")
			print(args.checkpoint_dir)
		else:
			print(" [!] Load failed...")

		classify_acc_sum = 0
		self.x_tst, self.y_tst = self.mnist.test.next_batch(10000)
		self.x_tst = np.reshape(self.x_tst, (10000, (self.image_size ** 2) * self.c_dim))

		for i in range(10000):
			print ('~~~~~~~~~Testing the %d th images~~~~~~~~~' % i)
			x = np.expand_dims(self.x_tst[i], axis=0)
			y_pred = self.sess.run(self.pred_label, feed_dict={self.input_image: x})
			classify_acc_sum_test = self.sess.run(self.train_acc_sum,\
			                        feed_dict={self.input_image: x, self.real_label: np.expand_dims(self.y_tst[i],axis=0)})
			classify_acc_sum = classify_acc_sum + classify_acc_sum_test

		classify_acc_rate_test = classify_acc_sum / 10000.0
		print(classify_acc_rate_test)
		with open(os.path.join(self.result_dir, self.model_name, 'test_result.txt'),'w') as f_result:
			f_result.write('The test classificaton accuracy is %4.4f' % classify_acc_rate_test)


	def save(self, checkpoint_dir, model_name, step):

		checkpoint_dir = os.path.join(checkpoint_dir, model_name)

		if not os.path.exists(checkpoint_dir):
			os.makedirs(checkpoint_dir)

		self.saver.save(self.sess, os.path.join(checkpoint_dir, model_name), global_step=step)


class LeNet_5(object):

	def __init__(self, sess, args):
		super(LeNet_5, self).__init__()

		self.dataset_dir = args.dataset_dir
		self.checkpoint_dir = args.checkpoint_dir
		self.result_dir = args.result_dir
		self.lenet_5 = lenet_5
		self.model_name = 'lenet5_classifier'

		self.sess = sess
		self.lr = args.lr

		if args.data_name == 'mnist':
			self.image_size = 32
			self.c_dim = 1
			self.mnist = utils.get_mnist_data(os.path.join(self.dataset_dir, args.data_name), one_hot=True)
			self.batch_size = args.batch_size
			self.vali_batch_size = args.vali_batch_size

		self._build_model()
		self.saver = tf.train.Saver()


	def _build_model(self):

		self.input_image = tf.placeholder(tf.float32, [None, self.image_size, self.image_size, self.c_dim], name='input_image')
		self.real_label = tf.placeholder(tf.float32, [None, 10], name='real_label')

		self.feat1, self.feat3, self.logits, self.pred_label = self.lenet_5(self.input_image, tf.nn.relu, None)
		self.feat1_avg = tf.reduce_mean(self.feat1, axis=-1)
		self.feat3_avg = tf.reduce_mean(self.feat3, axis=-1)

		self.loss = tf.reduce_mean(tf.losses.softmax_cross_entropy(self.real_label, self.logits))
		self.train_op = tf.train.AdamOptimizer(self.lr).minimize(self.loss)

		self.train_acc_sum = tf.reduce_sum(tf.cast(tf.equal(tf.argmax(self.pred_label, axis=1),tf.argmax(self.real_label, axis=1)),tf.int64))

		t_vars = tf.trainable_variables()
		self.t_vars = [var for var in t_vars]
		for var in t_vars: print(var.name)


	def train(self, args):

		init_op = tf.global_variables_initializer()
		self.sess.run(init_op)

		start_time = time.time()

		for i in range(args.max_iteration):
			self.x_trn, self.y_trn = self.mnist.train.next_batch(self.batch_size)
			self.x_trn = np.reshape(self.x_trn,[-1, 28, 28, 1])
			self.x_trn = np.pad(self.x_trn, ((0,),(2,),(2,),(0,)), mode='edge')
			print(self.x_trn.shape)
			loss, _ = self.sess.run([self.loss, self.train_op], \
				                     feed_dict={self.input_image: self.x_trn, self.real_label: self.y_trn})
			classify_sum = self.sess.run(self.train_acc_sum, \
				                     feed_dict={self.input_image: self.x_trn, self.real_label: self.y_trn})
			classify_acc = classify_sum/self.batch_size
			print("Iter: [%5d] loss:[%4.4f] acc:[%g] time: %4.4f" % (i, loss, classify_acc, time.time()-start_time))

			if np.mod(i, args.print_freq) == 0:
				valid_acc_rate, valid_loss = self.sample()
				print('~~~~~~~~~~~~~~Valid loss is %4.4f, and classify acc is %4.4f'%(valid_loss, valid_acc_rate))

				if not os.path.exists(os.path.join(self.result_dir, self.model_name)):
					os.makedirs(os.path.join(self.result_dir, self.model_name))

				with open(os.path.join(self.result_dir, self.model_name,\
				          'training_loss_'+self.model_name+'.txt'), 'a') as f1:
					f1.write('%d \t %4.4f \n' % (i, loss))

				with open(os.path.join(self.result_dir, self.model_name,\
					      'training_acc_'+self.model_name+'.txt'), 'a') as f2:
					f2.write('%d \t %4.4f \n' % (i, classify_acc))

				with open(os.path.join(self.result_dir, self.model_name,\
					      'validation_loss_'+self.model_name+'.txt'), 'a') as f3:
					f3.write('%d \t %4.4f \n' % (i,valid_loss))

				with open(os.path.join(self.result_dir, self.model_name,\
					      'validation_acc_'+self.model_name+'.txt'), 'a') as f4:
					f4.write('%d \t %4.4f \n' % (i,valid_acc_rate))

			if np.mod(i, args.save_freq) == 0:
				self.save(self.checkpoint_dir, self.model_name, i)


	def sample(self):
		self.x_tst, self.y_tst = self.mnist.test.next_batch(self.vali_batch_size)
		self.x_tst = np.reshape(self.x_tst,[-1, 28, 28, 1])
		self.x_tst = np.pad(self.x_tst, ((0,),(2,),(2,),(0,)), mode='edge')
		valid_loss, valid_acc_sum = self.sess.run([self.loss, self.train_acc_sum],\
						                           feed_dict={self.input_image: self.x_tst, self.real_label: self.y_tst})
		valid_acc_rate = valid_acc_sum / self.vali_batch_size

		return valid_acc_rate, valid_loss


	def load(self, checkpoint_dir, model_name):

		checkpoint_dir = os.path.join(checkpoint_dir, model_name)

		print('****************Loading the Saved Model from****************')
		print(checkpoint_dir)

		ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
		if ckpt and ckpt.model_checkpoint_path:
			ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
			self.saver.restore(self.sess, os.path.join(checkpoint_dir, ckpt_name))
			return True
		else:
			return False


	def test(self, args):

		init_op = tf.global_variables_initializer()
		self.sess.run(init_op)

		if self.load(args.checkpoint_dir, self.model_name):
			print(" [*] Load SUCCESS, the dir is")
			print(os.path.join(args.checkpoint_dir, self.model_name))
		else:
			print(" [!] Load failed...")

		classify_acc_sum = 0
		self.x_tst, self.y_tst = self.mnist.test.next_batch(10000)
		self.x_tst = np.reshape(self.x_tst,[-1, 28, 28, 1])
		self.x_tst = np.pad(self.x_tst, ((0,),(2,),(2,),(0,)), mode='edge')

		for k in range(10):
			save_dir = os.path.join(self.result_dir, self.model_name, 'feature_1', str(k))
			if not os.path.exists(save_dir):
				os.makedirs(save_dir)
			save_dir = os.path.join(self.result_dir, self.model_name, 'feature_3', str(k))
			if not os.path.exists(save_dir):
				os.makedirs(save_dir)

		for i in range(10000):
			print ('~~~~~~~~~Testing the %d th images~~~~~~~~~' % i)
			x = np.expand_dims(self.x_tst[i], axis=0)
			y = np.argmax(self.y_tst[i])

			y_pred = self.sess.run(self.pred_label, feed_dict={self.input_image: x})
			classify_acc_sum_test = self.sess.run(self.train_acc_sum,\
			                        feed_dict={self.input_image: x, self.real_label: np.expand_dims(self.y_tst[i],axis=0)})
			classify_acc_sum = classify_acc_sum + classify_acc_sum_test

			self.feature_1 = self.sess.run(self.feat1_avg, feed_dict={self.input_image: x})
			self.feature_1 = np.squeeze(self.feature_1)
			utils.image_save(i, os.path.join(self.result_dir, self.model_name, 'feature_1', str(y)), self.feature_1)

			self.feature_3 = self.sess.run(self.feat3_avg, feed_dict={self.input_image: x})
			self.feature_3 = np.squeeze(self.feature_3)
			utils.image_save(i, os.path.join(self.result_dir, self.model_name, 'feature_3', str(y)), self.feature_3)

		classify_acc_rate_test = classify_acc_sum / 10000.0
		print(classify_acc_rate_test)
		with open(os.path.join(self.result_dir, self.model_name, 'test_result.txt'),'w') as f_result:
			f_result.write('The test classificaton accuracy is %4.4f' % classify_acc_rate_test)


	def save(self, checkpoint_dir, model_name, step):

		checkpoint_dir = os.path.join(checkpoint_dir, model_name)

		if not os.path.exists(checkpoint_dir):
			os.makedirs(checkpoint_dir)

		self.saver.save(self.sess, os.path.join(checkpoint_dir, model_name), global_step=step)