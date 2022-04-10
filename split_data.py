import os
import random

p = './data'
img_lab = os.path.join(p, 'img_lab')

imgs = os.listdir(img_lab)

imgs = [i for i in imgs if 'sat' in i]

random.shuffle(imgs)
train_imgs = imgs[:int(len(imgs)*0.7)]
val_imgs = imgs[int(len(imgs)*0.7):int(len(imgs)*0.9)]
test_imgs = imgs[int(len(imgs)*0.9):]

f_train = open(os.path.join(p, 'train.txt'), 'w')
for i in train_imgs:
    f_train.writelines(i+'\n')

f_val = open(os.path.join(p, 'val.txt'), 'w')
for i in val_imgs:
    f_val.writelines(i+'\n')

f_test = open(os.path.join(p, 'test.txt'), 'w')
for i in test_imgs:
    f_test.writelines(i+'\n')
