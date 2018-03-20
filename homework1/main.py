"""
Author: Maosen Li
"""
import argparse
import os
import tensorflow as tf
tf.set_random_seed(19)
from model import EmotionClassifier


parser = argparse.ArgumentParser(description='')
parser.add_argument('--dataset_dir', dest='dataset_dir', default='./train_test', help='path of the dataset')
parser.add_argument('--train_set', dest='train_set', default='train_data.mat', help='training data')
parser.add_argument('--train_label', dest='train_label', default='train_label.mat', help='training label')
parser.add_argument('--test_set', dest='test_set', default='test_data.mat', help='testing data')
parser.add_argument('--test_label', dest='test_label', default='test_label.mat', help='testing label')
parser.add_argument('--hidden_dim', dest='hidden_dim', type=int, default=3, help='dimension of hidden layer')
parser.add_argument('--batch_size', dest='batch_size', type=int, default=64, help='data in one mini-batch')
parser.add_argument('--lr', dest='lr', type=float, default=0.0002, help='initial learning rate')
parser.add_argument('--beta1', dest='beta1', type=float, default=0.9, help='1st hyperparam of Adam')
parser.add_argument('--phase', dest='phase', default='train', help='train, test')
parser.add_argument('--epoch', dest='epoch', type=int, default=50, help='# of epoch')
parser.add_argument('--epoch_step', dest='epoch_step', type=int, default=100, help='# of epoch to decay lr')
parser.add_argument('--save_freq', dest='save_freq', type=int, default=1, help='save a model every save_freq iterations')
parser.add_argument('--print_freq', dest='print_freq', type=int, default=1, help='print the debug information every print_freq iterations')
parser.add_argument('--continue_train', dest='continue_train', type=bool, default=False, help='if continue training, load the latest model: 1: true, 0: false')
parser.add_argument('--checkpoint_dir', dest='checkpoint_dir', default='./checkpoint', help='models are saved here')
parser.add_argument('--result_dir', dest='result_dir', default='./result', help='result of the model testing')
parser.add_argument('--shuffle', dest='shuffle', type=bool, default=True, help='Shuffle the data set')

args = parser.parse_args()


def main(_):
    if not os.path.exists(args.checkpoint_dir):
        os.makedirs(args.checkpoint_dir)
    if not os.path.exists(args.result_dir):
        os.makedirs(args.result_dir)

    train_data = os.path.join(args.dataset_dir, args.train_set)
    train_label = os.path.join(args.dataset_dir, args.train_label)
    test_data = os.path.join(args.dataset_dir, args.test_set)
    test_label = os.path.join(args.dataset_dir, args.test_label)

    tfconfig = tf.ConfigProto(allow_soft_placement=True)
    tfconfig.gpu_options.allow_growth = True
    with tf.Session(config=tfconfig) as sess:
        model = EmotionClassifier(sess, args)
        model.train(args, train_data, train_label, test_data, test_label) if args.phase == 'train' \
            else model.test(args, test_data, test_label)

if __name__ == '__main__':
    tf.app.run()