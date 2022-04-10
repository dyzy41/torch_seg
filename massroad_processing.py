import cv2
import os
import numpy as np
from PIL import Image
import shutil
import tqdm
from config import *

def judge_img(exp_p, rate):
    img = np.asarray(Image.open(exp_p).convert('L'))
    shape = img.shape
    img = np.where(img == 255, 1, 0).flatten()
    if np.sum(img) > shape[0]*shape[1]*rate:
        return True
    else:
        return False


if __name__ == '__main__':
    src_root = root_data
    os.mkdir(os.path.join(src_root, 'train0'))
    os.mkdir(os.path.join(src_root, 'train_labels0'))
    train_imgs = os.listdir(os.path.join(src_root, 'train'))
    train_labs = os.listdir(os.path.join(src_root, 'train_labels'))
    for i in tqdm.tqdm(range(len(train_imgs))):
        if judge_img(os.path.join(os.path.join(src_root, 'train', train_imgs[i])), 0.1):
            shutil.move(os.path.join(os.path.join(src_root, 'train_labels', train_imgs[i])),
                        os.path.join(os.path.join(src_root, 'train_labels0', train_imgs[i])))
            shutil.move(os.path.join(os.path.join(src_root, 'train', train_imgs[i])), os.path.join(os.path.join(src_root, 'train0', train_imgs[i])))
    
    train_imgs = os.listdir(os.path.join(src_root, 'train'))
    train_labs = os.listdir(os.path.join(src_root, 'train_labels'))
    train_imgs = sorted(train_imgs)
    train_labs = sorted(train_labs)

    val_imgs = train_imgs[int(len(train_imgs)*0.8):]
    val_labs = train_labs[int(len(train_labs) * 0.8):]
    for i in tqdm.tqdm(range(len(val_imgs))):
        shutil.move(os.path.join(os.path.join(src_root, 'train_labels', val_imgs[i])),
                    os.path.join(os.path.join(src_root, 'val_labels', val_imgs[i])))
        shutil.move(os.path.join(os.path.join(src_root, 'train', val_imgs[i])), os.path.join(os.path.join(src_root, 'val', val_imgs[i])))

    print('ok')