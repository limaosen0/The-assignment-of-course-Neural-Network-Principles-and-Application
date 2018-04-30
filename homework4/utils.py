"""
Author: Maosen Li (017034910074), Shanghai Jiao Tong University
"""

import numpy as np
import os

import tensorflow as tf


def get_data_sep(data_dir, npz_name, film_id):

	zip_data = np.load(os.path.join(data_dir,(npz_name+'.npz')))
	files_in_zip = zip_data.keys()
	zip_data_T = np.transpose(zip_data[files_in_zip[film_id]],(1,2,0))
	sample_num = zip_data_T.shape[0]
	zip_data_rsh = np.reshape(zip_data_T,(sample_num,310))

	return zip_data_rsh, sample_num


def get_data_all(data_dir, film_id):

	zip_data_1 = np.load(os.path.join(data_dir,'01.npz'))
	zip_data_2 = np.load(os.path.join(data_dir,'02.npz'))
	zip_data_3 = np.load(os.path.join(data_dir,'03.npz'))

	files_in_zip_1 = zip_data_1.keys()
	zip_data_T_1 = np.transpose(zip_data_1[files_in_zip_1[film_id]],(1,2,0))
	sample_num_1 = zip_data_T_1.shape[0]
	zip_data_rsh_1 = np.reshape(zip_data_T_1,(sample_num_1,310))

	files_in_zip_2 = zip_data_2.keys()
	zip_data_T_2 = np.transpose(zip_data_2[files_in_zip_2[film_id]],(1,2,0))
	sample_num_2 = zip_data_T_2.shape[0]
	zip_data_rsh_2 = np.reshape(zip_data_T_2,(sample_num_2,310))

	files_in_zip_3 = zip_data_3.keys()
	zip_data_T_3 = np.transpose(zip_data_3[files_in_zip_3[film_id]],(1,2,0))
	sample_num_3 = zip_data_T_3.shape[0]
	zip_data_rsh_3 = np.reshape(zip_data_T_3,(sample_num_3,310))

	return zip_data_rsh_1, zip_data_rsh_2, zip_data_rsh_3, sample_num_1, sample_num_2, sample_num_3


def get_label(data_dir):

	data_label = np.load(os.path.join(data_dir,'label.npy'))

	return data_label


def get_label_onehot(data_dir):

	data_label = get_label(data_dir)
	onehot_label = []

	for i in range(len(data_label)):
		if data_label[i] == 0:
			onehot_label.append([1,0,0])
		elif data_label[i] == 1:
			onehot_label.append([0,1,0])
		else:
			onehot_label.append([0,0,1])

	return onehot_label


def get_clip_dataset_sep(data_dir, npz_name, clip_size, shuffle=False):

	clip_dataset_trn = []
	clip_dataset_tst = []

	train_film_num = 9
	test_film_num = 6

	for i in range(0, train_film_num):

		data_rsh, sample_num_trn = get_data_sep(data_dir, npz_name, i)
		onehot_label = get_label_onehot(data_dir)[i]

		for j in range(sample_num_trn-clip_size+1):
			random_start_point = j
			random_end_point = random_start_point + clip_size
			data_clip = data_rsh[random_start_point:random_end_point]
			data_and_label = (data_clip, onehot_label)
			clip_dataset_trn.append(data_and_label)

	for k in range(train_film_num, train_film_num+test_film_num):

		data_rsh, sample_num_tst = get_data_sep(data_dir, npz_name, k)
		onehot_label = get_label_onehot(data_dir)[k]

		for l in range(sample_num_tst-clip_size):
			random_start_point = l
			random_end_point = random_start_point + clip_size
			data_clip = data_rsh[random_start_point:random_end_point]
			data_and_label = (data_clip, onehot_label)
			clip_dataset_tst.append(data_and_label)

	if shuffle:
		np.random.shuffle(clip_dataset_trn)
		np.random.shuffle(clip_dataset_tst)

	return clip_dataset_trn, clip_dataset_tst


def get_clip_dataset_all(data_dir, clip_size, shuffle=False):

	clip_dataset_trn = []
	clip_dataset_tst = []

	train_film_num = 9
	test_film_num = 6

	for i in range(0, train_film_num):

		data_rsh_1, data_rsh_2, data_rsh_3, sample_num_trn_1, sample_num_trn_2, sample_num_trn_3 = get_data_all(data_dir, i)
		onehot_label = get_label_onehot(data_dir)[i]

		for a in range(sample_num_trn_1-clip_size+1):
			random_start_point = a
			random_end_point = random_start_point + clip_size
			data_clip = data_rsh_1[random_start_point:random_end_point]
			data_and_label = (data_clip, onehot_label)
			clip_dataset_trn.append(data_and_label)

		for b in range(sample_num_trn_2-clip_size+1):
			random_start_point = b
			random_end_point = random_start_point + clip_size
			data_clip = data_rsh_2[random_start_point:random_end_point]
			data_and_label = (data_clip, onehot_label)
			clip_dataset_trn.append(data_and_label)

		for c in range(sample_num_trn_3-clip_size+1):
			random_start_point = c
			random_end_point = random_start_point + clip_size
			data_clip = data_rsh_3[random_start_point:random_end_point]
			data_and_label = (data_clip, onehot_label)
			clip_dataset_trn.append(data_and_label)

	for j in range(train_film_num, train_film_num+test_film_num):

		data_rsh_1, data_rsh_2, data_rsh_3, sample_num_tst_1, sample_num_tst_2, sample_num_tst_3 = get_data_all(data_dir, j)
		onehot_label = get_label_onehot(data_dir)[j]

		for a in range(sample_num_tst_1-clip_size):
			random_start_point = a
			random_end_point = random_start_point + clip_size
			data_clip = data_rsh_1[random_start_point:random_end_point]
			data_and_label = (data_clip, onehot_label)
			clip_dataset_tst.append(data_and_label)

		for b in range(sample_num_tst_2-clip_size):
			random_start_point = b
			random_end_point = random_start_point + clip_size
			data_clip = data_rsh_2[random_start_point:random_end_point]
			data_and_label = (data_clip, onehot_label)
			clip_dataset_tst.append(data_and_label)

		for c in range(sample_num_tst_3-clip_size):
			random_start_point = c
			random_end_point = random_start_point + clip_size
			data_clip = data_rsh_3[random_start_point:random_end_point]
			data_and_label = (data_clip, onehot_label)
			clip_dataset_tst.append(data_and_label)

	if shuffle:
		np.random.shuffle(clip_dataset_trn)
		np.random.shuffle(clip_dataset_tst)

	return clip_dataset_trn, clip_dataset_tst


def softmax(x, axis=None, name=None, dim=None):

	return tf.nn.softmax(x, axis=axis, name=name, dim=dim)