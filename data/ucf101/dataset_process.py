import os
import pdb

output = []
files_input = ["ucf_test1.txt","ucf_train1.txt"]
files_output = ['val_ucf.txt','train_ucf.txt']
for (filename_input, filename_output) in zip(files_input, files_output):
    with open(filename_input) as f:
        lines = f.readlines()
    folders = []
    idx_categories = []
    for line in lines:
        line = line.rstrip()
        items = line.split(' ')
        curFolder = line[0]
        num = len(os.listdir(curFolder))
        curIDX = line[1]
        output.append('%s %s %s'%(curFolder, num, curIDX))
        print('%s %s %s'%(curFolder, num, curIDX))
    with open(filename_output,'w') as f:
        f.write('\n'.join(output))
