import os
import random
import sys

src_path = '/home/dsj/torch_seg_debug/dataset/city'

state = 'val'

imgs = os.listdir(os.path.join(src_path, 'images'))


with open('{}_list.txt'.format(state), 'w') as ff:
    for name in imgs:
        cur_info = '{}  {}\n'.format(os.path.join(src_path, 'images', name), os.path.join(src_path, 'masks', name.split('.')[0]+'.png'))
        ff.writelines(cur_info)

