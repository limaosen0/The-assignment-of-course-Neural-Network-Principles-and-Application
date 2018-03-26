from sklearn.svm import SVC
from sklearn import preprocessing
from sklearn.externals import joblib
import numpy as np
import argparse
import cv2
import os
import utils


def model_q1(data_path, save_path, train_data, train_label, test_data, test_label, C, iter_time, print_predict=False, PCA_visualize=False, tSNE_visualize=False):

	if not os.path.exists(save_path):
		os.mkdir(save_path)

	training_data = utils.load_data(data_path, train_data)
	testing_data = utils.load_data(data_path, test_data)
	training_label = utils.load_label(data_path, train_label)
	testing_label = utils.load_label(data_path, test_label)

	training_label_1 = utils.load_label(data_path, train_label, mode='1r')
	training_label_2 = utils.load_label(data_path, train_label, mode='2r')
	training_label_3 = utils.load_label(data_path, train_label, mode='3r')

	if PCA_visualize == True:
		utils.visual_2D_PCA(training_data, training_label, save_path)
	if tSNE_visualize == True:
		utils.visual_2D_tSNE(training_data, training_label, save_path)

	scaler = preprocessing.StandardScaler().fit(training_data)
	training_data = scaler.transform(training_data)
	testing_data = scaler.transform(testing_data)

	model = SVC(C=C, max_iter=iter_time, class_weight='balanced')
	model_1r = SVC(C=C, max_iter=iter_time, class_weight='balanced')
	model_2r = SVC(C=C, max_iter=iter_time, class_weight='balanced')
	model_3r = SVC(C=C, max_iter=iter_time, class_weight='balanced')

	model.fit(training_data, training_label)
	model_1r.fit(training_data, training_label_1)
	model_2r.fit(training_data, training_label_2)
	model_3r.fit(training_data, training_label_3)

	acc_multi = model.score(testing_data, testing_label)

	y_predict = []
	right_predict = 0.0
	for i in range(len(testing_data)):
		y1 = model_1r.predict([testing_data[i]])
		y2 = model_2r.predict([testing_data[i]])
		y3 = model_3r.predict([testing_data[i]])
		y_possible = np.where(np.array([y1, y2, y3])==1)[0]
		if len(y_possible) > 0:
			y_idx = np.random.randint(low=0, high=len(y_possible))
			y = y_possible[y_idx]+1;
		else:
			y = np.random.randint(low=1, high=4)
		y_predict.append(y)
		if y == testing_label[i][0]:
			right_predict = right_predict + 1.0

	acc = right_predict*1.0/len(testing_data)

	joblib.dump(model_1r, os.path.join(save_path, 'svm_model_1r.m'))
	joblib.dump(model_2r, os.path.join(save_path, 'svm_model_2r.m'))
	joblib.dump(model_3r, os.path.join(save_path, 'svm_model_3r.m'))

	if print_predict == True:
		print y_predict

	f1 = open(os.path.join(save_path, 'problem_1.txt'),'w')
	f1.write('If we use multi-classifying SVM in sklearn, the classification accuracy is: %g;\n'%(acc_multi))
	f1.write('If we use our 1 vs Rest model by sklearn, the classification accuracy is: %g.'%(acc))
	f1.close()

	return acc_multi, acc


def model_q2(data_path, save_path, train_data, train_label, test_data, test_label, C, iter_time, separate_type, print_predict=False, PCA_visualize=False, tSNE_visualize=False):

	if not os.path.exists(save_path):
		os.mkdir(save_path)

	training_data = utils.load_data(data_path, train_data)
	training_label = utils.load_label(data_path, train_label)
	testing_data = utils.load_data(data_path, test_data)
	testing_label = utils.load_label(data_path, test_label)

	if separate_type == 'random':
		trn_1_r1_data, trn_1_r1_label, trn_1_r2_data, trn_1_r2_label = utils.separate_data_random(data_path, train_data, train_label, mode='1r')
		trn_2_r1_data, trn_2_r1_label, trn_2_r2_data, trn_2_r2_label = utils.separate_data_random(data_path, train_data, train_label, mode='2r')
		trn_3_r1_data, trn_3_r1_label, trn_3_r2_data, trn_3_r2_label = utils.separate_data_random(data_path, train_data, train_label, mode='3r')
	elif separate_type == 'prior':
		trn_1_r1_data, trn_1_r1_label, trn_1_r2_data, trn_1_r2_label = utils.separate_data_prior(data_path, train_data, train_label, mode='1r')
		trn_2_r1_data, trn_2_r1_label, trn_2_r2_data, trn_2_r2_label = utils.separate_data_prior(data_path, train_data, train_label, mode='2r')
		trn_3_r1_data, trn_3_r1_label, trn_3_r2_data, trn_3_r2_label = utils.separate_data_prior(data_path, train_data, train_label, mode='3r')

	if PCA_visualize == True:
		utils.visual_2D_PCA(training_data, training_label, save_path)
	if tSNE_visualize == True:
		utils.visual_2D_tSNE(training_data, training_label, save_path)

	scaler = preprocessing.StandardScaler().fit(training_data)
	trn_1_r1_data = scaler.transform(trn_1_r1_data)
	trn_1_r2_data = scaler.transform(trn_1_r2_data)
	trn_2_r1_data = scaler.transform(trn_2_r1_data)
	trn_2_r2_data = scaler.transform(trn_2_r2_data)
	trn_3_r1_data = scaler.transform(trn_3_r1_data)
	trn_3_r2_data = scaler.transform(trn_3_r2_data)
	testing_data = scaler.transform(testing_data)

	model_1_r1 = SVC(C=C, max_iter=iter_time, class_weight='balanced')
	model_1_r2 = SVC(C=C, max_iter=iter_time, class_weight='balanced')
	model_2_r1 = SVC(C=C, max_iter=iter_time, class_weight='balanced')
	model_2_r2 = SVC(C=C, max_iter=iter_time, class_weight='balanced')
	model_3_r1 = SVC(C=C, max_iter=iter_time, class_weight='balanced')
	model_3_r2 = SVC(C=C, max_iter=iter_time, class_weight='balanced')

	model_1_r1.fit(trn_1_r1_data, trn_1_r1_label)
	model_1_r2.fit(trn_1_r2_data, trn_1_r2_label)
	model_2_r1.fit(trn_2_r1_data, trn_2_r1_label)
	model_2_r2.fit(trn_2_r2_data, trn_2_r2_label)
	model_3_r1.fit(trn_3_r1_data, trn_3_r1_label)
	model_3_r2.fit(trn_3_r2_data, trn_3_r2_label)

	y_predict = []
	right_predict = 0.0
	for i in range(len(testing_data)):
		y1_r1 = int(model_1_r1.predict([testing_data[i]])[0])
		y1_r2 = int(model_1_r2.predict([testing_data[i]])[0])
		y2_r1 = int(model_2_r1.predict([testing_data[i]])[0])
		y2_r2 = int(model_2_r2.predict([testing_data[i]])[0])
		y3_r1 = int(model_3_r1.predict([testing_data[i]])[0])
		y3_r2 = int(model_3_r2.predict([testing_data[i]])[0])

		predict_1 = [[y1_r1, y1_r2],[y2_r1, y2_r2],[y3_r1, y3_r2]]
		predict_2 = [y1_r1 and y1_r2, y2_r1 and y2_r2, y3_r1 and y3_r2]

		y_possible = np.where(np.array(predict_2)==1)[0]

		if len(y_possible) > 0:
			y_idx = np.random.randint(low=0, high=len(y_possible))
			y = y_possible[y_idx]+1;
		else:
			y = np.random.randint(low=1, high=4)
		y_predict.append(y)
		if y == testing_label[i][0]:
			right_predict = right_predict + 1.0

	acc = right_predict*1.0/len(testing_data)

	joblib.dump(model_1_r1, os.path.join(save_path, 'svm_model_1_r1.m'))
	joblib.dump(model_1_r2, os.path.join(save_path, 'svm_model_1_r2.m'))
	joblib.dump(model_2_r1, os.path.join(save_path, 'svm_model_2_r1.m'))
	joblib.dump(model_2_r2, os.path.join(save_path, 'svm_model_2_r2.m'))
	joblib.dump(model_3_r1, os.path.join(save_path, 'svm_model_3_r1.m'))
	joblib.dump(model_3_r2, os.path.join(save_path, 'svm_model_3_r2.m'))

	if print_predict == True:
		print y_predict

	f1 = open(os.path.join(save_path, 'problem_2.txt'),'a')
	f1.write('If we use our Part vs Part model by sklearn, and the separate type is %s, the classification accuracy is: %g.\n'%(separate_type, acc))
	f1.close()

	return acc