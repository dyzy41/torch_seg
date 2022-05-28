import os
import random
import sys

src_path = r'E:\0Epaper\whub\whub'

state = 'val'

image_path = os.path.join(src_path, '{}'.format(state), 'image')
label_path = os.path.join(src_path, '{}'.format(state), 'label')

images = os.listdir(image_path)
random.shuffle(images)

with open('{}_list.txt'.format(state), 'w') as ff:
    for name in images:
        cur_info = '{}  {}\n'.format(os.path.join(image_path, name), os.path.join(label_path, name))
        ff.writelines(cur_info)

