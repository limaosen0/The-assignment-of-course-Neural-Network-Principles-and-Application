"""
Author: Maosen Li
"""
import numpy as np
import os
import scipy.io


def data_load(input_data):

	data = scipy.io.loadmat(input_data)
	data_key = os.path.basename(input_data[:-4])
	data_value = data[data_key]
	data_shape = data_value.shape
	data_num = data_shape[0]
	data_dim = data_shape[1]
	if data_dim > 3:
		data_value = ((np.array(data_value)-12.9605)/(44.1745-12.9605)-0.5)*2

	return np.array(data_value), data_num, data_dim


def label_transform(input_label):

	input_label = input_label[0]
	
	if input_label == 1:
		vector_label = np.array([1, 0, 0])
	elif input_label == 0:
		vector_label = np.array([0, 1, 0])
	elif input_label == -1:
		vector_label = np.array([0, 0, 1])

	return vector_label


def data_label_combine(input_data, input_label):

	data_value, data_num, data_dim = data_load(input_data)
	label_value, label_num, label_dim = data_load(input_label)
	assert data_num == label_num

	data_and_label = []
	for i in range(data_num):
		vector_label = label_transform(label_value[i])
		data_label_pair = (data_value[i], vector_label)
		data_and_label.append(data_label_pair)

	return data_and_label
