import os
import pdb
with open("kinetics200_train_list") as f:
    lines = f.readlines()
new =[]
for line in lines:
    line = line.rstrip()
    line = line.split(' ')
    line0 = line[0].split('/')[1]+'.mp4'
    line1 = 'TBD'
    line2 = line[2]
    linenew = line0 + ' ' + line1 + ' ' + line2
    new.append(linenew)
with open("200_train",'w') as f:
    f.write('\n'.join(new))
