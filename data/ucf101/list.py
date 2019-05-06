import os
import csv
import collections
from collections import OrderedDict

train_root = '/data0/action_database/moments_mini/training'
val_root = '/data0/action_database/moments_mini/validation'

cls_ind_dict = OrderedDict()
with open('moments_categories.txt', 'rb') as f:
	lines = f.readlines()
	for line in lines:
		items = line.strip().split(',')
		cls_name = items[0]
		cls_index = items[1]
		cls_ind_dict[cls_name] = cls_index

with open('trainingSet.csv', 'rb') as f_tr:
	with open('validationSet.csv', 'rb') as f_va:
		with open('moments_mini_train.txt', 'wt') as f_tr_dst:
			with open('moments_mini_val.txt', 'wt') as f_va_dst:
				reader_tr = csv.reader(f_tr)
				reader_va = csv.reader(f_va)
				for line in reader_tr:
					vid_name = line[0].split('/')[-1].split('.')[0]
					label = cls_ind_dict[line[1]]
					vid_path = os.path.join(train_root, vid_name)
					if os.path.isdir(vid_path):
						num_frame = len(os.listdir(vid_path))
						if num_frame % 2 == 0 and num_frame > 80:
							f_tr_dst.write(' '.join(['{}/{}'.format('training', vid_name), str(num_frame/2), label]) + '\n')
				for line in reader_va:
					vid_name = line[0].split('/')[-1].split('.')[0]
					label = cls_ind_dict[line[1]]
					vid_path = os.path.join(val_root, vid_name)
					if os.path.isdir(vid_path):
						num_frame = len(os.listdir(vid_path))
						if num_frame % 2 == 0 and num_frame > 80:
							f_va_dst.write(' '.join(['{}/{}'.format('validation', vid_name), str(num_frame/2), label]) + '\n')
