import numpy as np
import os
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def load_data(path, file_name):

	raw_data = np.load(os.path.join(path, file_name))
	data = np.array(raw_data)
	return data


def load_label(path, file_name, mode=None):

	label = []
	raw_label = np.load(os.path.join(path, file_name))

	if mode == '1r':
		for i in range(len(raw_label)):
			if raw_label[i] == 1.0:
				label.append([1.0])
			elif raw_label[i] == 0.0:
				label.append([0.0])
			elif raw_label[i] == -1.0:
				label.append([0.0])

	elif mode == '2r':
		for i in range(len(raw_label)):
			if raw_label[i] == 1.0:
				label.append([0.0])
			elif raw_label[i] == 0.0:
				label.append([1.0])
			elif raw_label[i] == -1.0:
				label.append([0.0])

	elif mode == '3r':
		for i in range(len(raw_label)):
			if raw_label[i] == 1.0:
				label.append([0.0])
			elif raw_label[i] == 0.0:
				label.append([0.0])
			elif raw_label[i] == -1.0:
				label.append([1.0])

	else:
		for i in range(len(raw_label)):
			if raw_label[i] == 1.0:
				label.append([1.0])
			elif raw_label[i] == 0.0:
				label.append([2.0])
			elif raw_label[i] == -1.0:
				label.append([3.0])

	label = np.array(label)
	return label


def separate_data_random(path, data_file_name, label_file_name, mode):

	one_v_r1_data_raw = []
	one_v_r2_data_raw = []
	one_v_r1_label_raw = []
	one_v_r2_label_raw = []

	raw_data = np.load(os.path.join(path, data_file_name))
	raw_label = np.load(os.path.join(path, label_file_name))

	if mode == '1r':
		for i in range(len(raw_label)):
			p = np.random.rand(1)
			if raw_label[i] == 1.0:
				one_v_r1_data_raw.append(raw_data[i])
				one_v_r1_label_raw.append([1.0])
				one_v_r2_data_raw.append(raw_data[i])
				one_v_r2_label_raw.append([1.0])
			if raw_label[i] == 0.0:
				if p >= 0.5:
					one_v_r1_data_raw.append(raw_data[i])
					one_v_r1_label_raw.append([0.0])
				else:
					one_v_r2_data_raw.append(raw_data[i])
					one_v_r2_label_raw.append([0.0])
			if raw_label[i] == -1.0:
				if p >= 0.5:
					one_v_r1_data_raw.append(raw_data[i])
					one_v_r1_label_raw.append([0.0])
				else:
					one_v_r2_data_raw.append(raw_data[i])
					one_v_r2_label_raw.append([0.0])

	if mode == '2r':
		for i in range(len(raw_label)):
			p = np.random.rand(1)
			if raw_label[i] == 0.0:
				one_v_r1_data_raw.append(raw_data[i])
				one_v_r1_label_raw.append([1.0])
				one_v_r2_data_raw.append(raw_data[i])
				one_v_r2_label_raw.append([1.0])
			if raw_label[i] == 1.0:
				if p >= 0.5:
					one_v_r1_data_raw.append(raw_data[i])
					one_v_r1_label_raw.append([0.0])
				else:
					one_v_r2_data_raw.append(raw_data[i])
					one_v_r2_label_raw.append([0.0])
			if raw_label[i] == -1.0:
				if p >= 0.5:
					one_v_r1_data_raw.append(raw_data[i])
					one_v_r1_label_raw.append([0.0])
				else:
					one_v_r2_data_raw.append(raw_data[i])
					one_v_r2_label_raw.append([0.0])

	if mode == '3r':
		for i in range(len(raw_label)):
			p = np.random.rand(1)
			if raw_label[i] == -1.0:
				one_v_r1_data_raw.append(raw_data[i])
				one_v_r1_label_raw.append([1.0])
				one_v_r2_data_raw.append(raw_data[i])
				one_v_r2_label_raw.append([1.0])
			if raw_label[i] == 1.0:
				if p >= 0.5:
					one_v_r1_data_raw.append(raw_data[i])
					one_v_r1_label_raw.append([0.0])
				else:
					one_v_r2_data_raw.append(raw_data[i])
					one_v_r2_label_raw.append([0.0])
			if raw_label[i] == 0.0:
				if p >= 0.5:
					one_v_r1_data_raw.append(raw_data[i])
					one_v_r1_label_raw.append([0.0])
				else:
					one_v_r2_data_raw.append(raw_data[i])
					one_v_r2_label_raw.append([0.0])

	one_v_r1_data = np.array(one_v_r1_data_raw)
	one_v_r2_data = np.array(one_v_r2_data_raw)
	one_v_r1_label = np.array(one_v_r1_label_raw)
	one_v_r2_label = np.array(one_v_r2_label_raw)

	return one_v_r1_data, one_v_r1_label, one_v_r2_data, one_v_r2_label
		

def separate_data_prior(path, data_file_name, label_file_name, mode):

	one_v_r1_data_raw = []
	one_v_r2_data_raw = []
	one_v_r1_label_raw = []
	one_v_r2_label_raw = []

	raw_data = np.load(os.path.join(path, data_file_name))
	raw_label = np.load(os.path.join(path, label_file_name))

	if mode == '1r':
		for i in range(len(raw_label)):
			if raw_label[i] == 1.0:
				one_v_r1_data_raw.append(raw_data[i])
				one_v_r1_label_raw.append([1.0])
				one_v_r2_data_raw.append(raw_data[i])
				one_v_r2_label_raw.append([1.0])
			if raw_label[i] == 0.0:
				one_v_r1_data_raw.append(raw_data[i])
				one_v_r1_label_raw.append([0.0])
			if raw_label[i] == -1.0:
				one_v_r2_data_raw.append(raw_data[i])
				one_v_r2_label_raw.append([0.0])

	if mode == '2r':
		for i in range(len(raw_label)):
			if raw_label[i] == 0.0:
				one_v_r1_data_raw.append(raw_data[i])
				one_v_r1_label_raw.append([1.0])
				one_v_r2_data_raw.append(raw_data[i])
				one_v_r2_label_raw.append([1.0])
			if raw_label[i] == 1.0:				
				one_v_r1_data_raw.append(raw_data[i])
				one_v_r1_label_raw.append([0.0])
			if raw_label[i] == -1.0:
				one_v_r2_data_raw.append(raw_data[i])
				one_v_r2_label_raw.append([0.0])

	if mode == '3r':
		for i in range(len(raw_label)):
			if raw_label[i] == -1.0:
				one_v_r1_data_raw.append(raw_data[i])
				one_v_r1_label_raw.append([1.0])
				one_v_r2_data_raw.append(raw_data[i])
				one_v_r2_label_raw.append([1.0])
			if raw_label[i] == 1.0:
				one_v_r1_data_raw.append(raw_data[i])
				one_v_r1_label_raw.append([0.0])
			if raw_label[i] == 0.0:
				one_v_r2_data_raw.append(raw_data[i])
				one_v_r2_label_raw.append([0.0])

	one_v_r1_data = np.array(one_v_r1_data_raw)
	one_v_r2_data = np.array(one_v_r2_data_raw)
	one_v_r1_label = np.array(one_v_r1_label_raw)
	one_v_r2_label = np.array(one_v_r2_label_raw)

	return one_v_r1_data, one_v_r1_label, one_v_r2_data, one_v_r2_label


def visual_2D_PCA(train_data, train_label, save_path):

	train_label = np.ravel(train_label)
	pca = PCA(n_components = 2)
	X_r = pca.fit(train_data).transform(train_data)

	plt.figure()
	colors = ['navy', 'turquoise', 'darkorange']
	lw = 2
	target_names = ['-1', '0', '1']

	for color, i, target_name in zip(colors, [3.0,2.0,1.0], target_names):
		plt.scatter(X_r[train_label == i, 0], X_r[train_label == i, 1], color=color, alpha=.8,  linewidths=lw, label=target_name)
	plt.legend(loc='best', shadow=False, scatterpoints=1)
	plt.title('PCA of Training Data')
	plt.show()
	plt.savefig(os.path.join(save_path, 'PCA_data.png'))


def visual_2D_tSNE(train_data, train_label, save_path):

	train_label = np.ravel(train_label)
	tsne = TSNE(n_components = 2)
	X_r = tsne.fit_transform(train_data)

	plt.figure()
	colors = ['navy', 'turquoise', 'darkorange']
	lw = 1
	target_names = ['-1', '0', '1']

	for color, i, target_name in zip(colors, [3.0,2.0,1.0], target_names):
		plt.scatter(X_r[train_label == i, 0], X_r[train_label == i, 1], color=color, alpha=.8,  linewidths=lw, label=target_name)
	plt.legend(loc='best', shadow=False, scatterpoints=1)
	plt.title('tSNE of Training Data')
	plt.show()
	plt.savefig(os.path.join(save_path, 'TSNE_data.png'))