import os
from shutil import *

dataset_name_list = ['01', '02', '03']
layers_num_list = [2, 3]
lstm_unit_list = [16, 32, 64, 128]
lstm_step_list = [8, 16, 32, 64]
batch_size_list = [128]

for batch_size in batch_size_list:
    for layers_num in layers_num_list:
        for lstm_unit in lstm_unit_list:
            for lstm_step in lstm_step_list:
            
                os.system('CUDA_VISIBLE_DEVICES=3 python main.py --dataset_name={} --layers_num={} --lstm_unit={} --lstm_step={} --batch_size={}'\
                          .format('03', layers_num, lstm_unit, lstm_step, batch_size))


