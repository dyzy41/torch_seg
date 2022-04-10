import os
from tools.utils import *
import random
import shutil



p = r'F:\0DL_datasets\WHU_building\3. The cropped image tiles and raster labels'
tgt = r'F:\0Fcode\code220303_seg\whub'

dirs = os.listdir(p)

for item in dirs:
    check_path(os.path.join(tgt, item))
    image_path = os.path.join(p, item, 'image')
    label_path = os.path.join(p, item, 'label')
    tgt_image_path = os.path.join(tgt, item, 'image')
    tgt_label_path = os.path.join(tgt, item, 'label')
    check_path(tgt_image_path)
    check_path(tgt_label_path)
    filenames = os.listdir(image_path)
    random.shuffle(filenames)
    filenames = filenames[:int(len(filenames)*0.4)]

    for name in filenames:
        shutil.copy(os.path.join(image_path, name), os.path.join(tgt_image_path, name))
        shutil.copy(os.path.join(label_path, name), os.path.join(tgt_label_path, name))

