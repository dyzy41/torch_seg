import os
import random
import sys

src_path = r'E:\dataset\smallcity'

state = 'val'

images = os.listdir(os.path.join(src_path, 'images'))

with open('{}_list.txt'.format(state), 'w') as ff:
    for name in images:
        cur_info = '{}  {}\n'.format(os.path.join(src_path, 'images', name), os.path.join(src_path, 'masks', name.replace('.jpg', '.png')))
        ff.writelines(cur_info)

