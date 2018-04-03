"""
Author: Maosen Li (017034910074), Shanghai Jiao Tong University
"""

import argparse
import os
import tensorflow as tf
tf.set_random_seed(19)
from model import NN_Classifier, LeNet_5


parser = argparse.ArgumentParser(description='')

parser.add_argument('--dataset_dir', dest='dataset_dir', default='./data', help='path of the dataset')
parser.add_argument('--data_name', dest='data_name', default='mnist', help='name of the dataset')
parser.add_argument('--save_freq', dest='save_freq', type=int, default=100, help='save model freq')
parser.add_argument('--print_freq', dest='print_freq', type=int, default=100, help='validation freq')
parser.add_argument('--checkpoint_dir', dest='checkpoint_dir', default='./checkpoint', help='models are saved here')
parser.add_argument('--result_dir', dest='result_dir', default='./result', help='result of the model testing')

parser.add_argument('--network_type', dest='network_type', default='fc', help='only choose "fc" or "cnn"')
parser.add_argument('--hidden_dim', dest='hidden_dim', type=int, default=1024, help='dimension of hidden layer')
parser.add_argument('--layers_num', dest='layers_num', type=int, default=5, help='how many layers in model')
parser.add_argument('--dropout', dest='dropout', type=bool, default=True, help='Using dropout in training')
parser.add_argument('--keep_prob', dest='keep_prob', type=float, default=0.5, help='prob of keeping activitation nodes')
parser.add_argument('--lr', dest='lr', type=float, default=0.0002, help='initial learning rate')
parser.add_argument('--beta1', dest='beta1', type=float, default=0.9, help='1st hyperparam of Adam')

parser.add_argument('--phase', dest='phase', default='train', help='train, test')
parser.add_argument('--max_iteration', dest='max_iteration', type=int, default=10001, help='max iteration step')
parser.add_argument('--batch_size', dest='batch_size', type=int, default=128, help='mini-batch size')
parser.add_argument('--vali_batch_size', dest='vali_batch_size', type=int, default=1000, help='mini-batch size to vali')

args = parser.parse_args()


def main(_):

	if not os.path.exists(args.checkpoint_dir):
		os.makedirs(args.checkpoint_dir)
	if not os.path.exists(args.result_dir):
		os.makedirs(args.result_dir)

	if args.network_type == 'fc':
		tfconfig = tf.ConfigProto(allow_soft_placement=True)
		tfconfig.gpu_options.allow_growth = True
		with tf.Session(config=tfconfig) as sess:
			model = NN_Classifier(sess, args)
			model.train(args) if args.phase == 'train' \
                  else model.test(args)
	elif args.network_type == 'cnn':
		tfconfig = tf.ConfigProto(allow_soft_placement=True)
		tfconfig.gpu_options.allow_growth = True
		with tf.Session(config=tfconfig) as sess:
			model = LeNet_5(sess, args)
			model.train(args) if args.phase == 'train' \
    		       else model.test(args)


if __name__ == '__main__':
	tf.app.run()