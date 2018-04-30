"""
Author: Maosen Li (017034910074), Shanghai Jiao Tong University
"""

from __future__ import division
import os
import time

import utils

import tensorflow as tf
import numpy as np


class 	LSTM_emotion(object):

	def __init__(self, sess, args):

		super(LSTM_emotion, self).__init__()

		self.problem_num = args.problem_num
		self.dataset_dir = args.dataset_dir
		self.dataset_name = args.dataset_name
		self.checkpoint_dir = args.checkpoint_dir
		self.result_dir = args.result_dir
		
		self.sess = sess
		self.lr = args.lr
		self.input_dim = args.input_dim
		self.output_dim = 3
		self.lstm_unit = args.lstm_unit
		self.lstm_step = args.lstm_step
		self.layers_num = args.layers_num

		self.shuffle = args.shuffle
		self.dropout = args.dropout
		self.keep_prob = args.keep_prob
		self.batch_size = args.batch_size

		if self.problem_num == '1':
			self.model_name = 'lstm_' + self.dataset_name + '_'\
		                          	  + str(self.layers_num) + '_'\
		                        	  + str(self.lstm_unit) + '_'\
		                        	  + str(self.lstm_step) + '_'\
		                        	  + str(self.batch_size)
		elif self.problem_num == '2':
			self.model_name = 'lstm_' + '04_all_data' + '_'\
		                          	  + str(self.layers_num) + '_'\
		                        	  + str(self.lstm_unit) + '_'\
		                        	  + str(self.lstm_step) + '_'\
		                        	  + str(self.batch_size)

		self._build_model()
		self.saver = tf.train.Saver()


	def _build_model(self):

		self.input_data = tf.placeholder(tf.float32, [None, self.lstm_step, self.input_dim], name='input_data')
		self.real_label = tf.placeholder(tf.float32, [None, self.output_dim], name='real_label')

		self.logits, self.pred_label = self.LSTM(self.input_data, self.lstm_unit, self.layers_num)

		self.xentropy = tf.losses.softmax_cross_entropy(onehot_labels=self.real_label, logits=self.logits)
		self.loss = tf.reduce_mean(self.xentropy, name='loss')
		self.train_op = tf.train.AdamOptimizer(self.lr).minimize(self.loss)

		self.train_acc_sum = tf.reduce_sum(tf.cast(tf.equal(tf.argmax(self.pred_label, axis=1),tf.argmax(self.real_label, axis=1)),tf.int64))

		t_vars = tf.trainable_variables()
		self.t_vars = [var for var in t_vars]
		for var in t_vars: print(var.name)

		if self.problem_num == '1':
			self.data_train, self.data_test = utils.get_clip_dataset_sep(self.dataset_dir,\
		                                                                 self.dataset_name,\
		                                                                 self.lstm_step,\
		                                                                 shuffle=self.shuffle)
		elif self.problem_num == '2':
			self.data_train, self.data_test = utils.get_clip_dataset_all(self.dataset_dir,\
		                                                                 self.lstm_step,\
		                                                                 shuffle=self.shuffle)

		print('~~~~~~~~~~The training number: %g, the testing number: %g.' % (len(self.data_train),len(self.data_test)))


	def LSTM(self, input_data, lstm_unit, layers_num):

		lstm_cells = [tf.contrib.rnn.BasicLSTMCell(num_units=self.lstm_unit)\
		              for layer in range(self.layers_num)]
		if self.dropout:
			cells_drop = [tf.contrib.rnn.DropoutWrapper(cell, input_keep_prob=self.keep_prob)\
		                  for cell in lstm_cells]
		                  
		self.multi_cell = tf.contrib.rnn.MultiRNNCell(cells_drop)
		self.outputs, self.states = tf.nn.dynamic_rnn(self.multi_cell,\
		                                              self.input_data,\
		                                              dtype=tf.float32)
		self.top_layer_h_state = self.states[-1][1]
		self.logits = tf.layers.dense(self.top_layer_h_state, self.output_dim, name="softmax")
		self.pred_label = utils.softmax(self.logits)

		return self.logits, self.pred_label


	def train(self, args):

		init_op = tf.global_variables_initializer()
		self.sess.run(init_op)

		iter_before_shuffle = len(self.data_train) // self.batch_size
		print('It can be seperate into %g mini-batches' % iter_before_shuffle)
		a = -1

		start_time = time.time()
		for i in range(args.max_iteration):

			a = a+1
			if a == (iter_before_shuffle):
				np.random.shuffle(self.data_train)
				a = 0

			self.data_batch_trn = self.data_train[a*self.batch_size:(a+1)*self.batch_size]
			self.x_trn = []
			self.y_trn = []
			for j in range(self.batch_size):
				x_ = self.data_batch_trn[j][0]
				self.x_trn.append(x_)
				y_ = self.data_batch_trn[j][1]
				self.y_trn.append(y_)

			self.x_trn = np.array(self.x_trn)
			self.y_trn = np.array(self.y_trn)

			loss, _ = self.sess.run([self.loss, self.train_op], \
				                     feed_dict={self.input_data: self.x_trn, self.real_label: self.y_trn})
			classify_sum = self.sess.run(self.train_acc_sum, \
				                     feed_dict={self.input_data: self.x_trn, self.real_label: self.y_trn})
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

		valid_acc_sum = 0
		for i in range(len(self.data_test)):
			self.data_tst = self.data_test[i]
			self.x_tst = np.expand_dims(np.array(self.data_tst[0]),axis=0)
			self.y_tst = np.expand_dims(np.array(self.data_tst[1]),axis=0)
			valid_loss, valid_acc = self.sess.run([self.loss, self.train_acc_sum],\
						                           	   feed_dict={self.input_data: self.x_tst, self.real_label: self.y_tst})
			valid_acc_sum = valid_acc_sum + valid_acc
		valid_acc_rate = valid_acc_sum / len(self.data_test)

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


	def save(self, checkpoint_dir, model_name, step):

		checkpoint_dir = os.path.join(checkpoint_dir, model_name)

		if not os.path.exists(checkpoint_dir):
			os.makedirs(checkpoint_dir)

		self.saver.save(self.sess, os.path.join(checkpoint_dir, model_name), global_step=step)