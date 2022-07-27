import os
import random
import shutil


src = r'F:\0Fcode\code220215_imgsegDL\dataset\data_slice'
filenames = os.listdir(os.path.join(src, 'image_train'))
filenames = [i.split('.')[0] for i in filenames]
random.shuffle(filenames)

val_files = filenames[int(len(filenames)*0.8):]
os.mkdir(os.path.join(src, 'image_val'))
os.mkdir(os.path.join(src, 'label_val'))
for item in val_files:
    shutil.move(os.path.join(src, 'image_train', item+'.jpg'), os.path.join(src, 'image_val', item+'.jpg'))
    shutil.move(os.path.join(src, 'label_train', item + '.png'), os.path.join(src, 'label_val', item + '.png'))
