import os
import random
import sys

src_path = r'E:\torch_seg\dataset\voc'

state = 'train'



images = open(os.path.join(src_path, '{}.txt'.format(state)), 'r').readlines()
random.shuffle(images)

with open('{}_list.txt'.format(state), 'w') as ff:
    for name in images:
        cur_info = '{}  {}\n'.format(os.path.join(src_path, 'cut_data/image_{}'.format('train'), name.strip()+'_1_1.jpg'), os.path.join(src_path, 'cut_data/label_{}'.format('train'), name.strip()+'_1_1.png'))
        ff.writelines(cur_info)

