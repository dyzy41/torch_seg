import warnings

warnings.filterwarnings('ignore')
import cv2
import numpy as np
import tqdm
import yimage
import os

# p = './data_slice_DRIVE/label_train'
# p = './CHASE_new/big_img'

p_lab = r'E:\0Ecode\code220329_stoneseg\dataset\test\label'
p_img = r'E:\0Ecode\code220329_stoneseg\dataset\test\image'
imgs = os.listdir(p_img)
imgs = [i.replace('.jpg', '.png') for i in imgs]

for i in tqdm.tqdm(range(len(imgs))):
    lab = cv2.imread(os.path.join(p_lab, imgs[i]), 0)
    if lab is None:
        print(imgs[i])




