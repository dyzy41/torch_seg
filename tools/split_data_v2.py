import os
import random
import shutil


src_path = r'E:\0Ecode\code220329_stoneseg\dataset\val'
tgt_path = r'E:\0Ecode\code220329_stoneseg\dataset\test'
os.mkdir(tgt_path)
tgt_img_path = os.path.join(tgt_path, 'image')
tgt_lab_path = os.path.join(tgt_path, 'label')
os.mkdir(tgt_img_path)
os.mkdir(tgt_lab_path)


img_names = os.listdir(os.path.join(src_path, 'image'))
random.shuffle(img_names)
val_imgs = img_names[:int(len(img_names)*0.33)]

for item in val_imgs:
    shutil.move(os.path.join(src_path, 'image', item), os.path.join(tgt_img_path, item))
    shutil.move(os.path.join(src_path, 'label', item.replace('.jpg', '.png')), os.path.join(tgt_lab_path, item.replace('.jpg', '.png')))
