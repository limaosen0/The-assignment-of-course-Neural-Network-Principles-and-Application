import os
from model import model_q1, model_q2
import utils

model_num = 2

data_path = './data'
save_path = './result'
model_name = 'model_'+str(model_num)
save_path = os.path.join(save_path, model_name)

C = 10.0
iter_time = 2000
separate_type = 'prior'

if model_num == 1:
	acc_multi, acc_ours = model_q1(data_path, save_path, 'train_data.npy', 'train_label.npy', 'test_data.npy', 'test_label.npy', C, iter_time, print_predict=False)
elif model_num == 2:
	acc_ours = model_q2(data_path, save_path, 'train_data.npy', 'train_label.npy', 'test_data.npy', 'test_label.npy', C, iter_time, separate_type, print_predict=False)