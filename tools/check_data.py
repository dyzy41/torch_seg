import os
import cv2
# import yimage
import numpy as np



p1 = 'dataset/label_data/image_train'
p2 = 'dataset/label_data/label_train'


files2 = os.listdir(p2)
files1 = os.listdir(p1)
id = 0
for item in files2:
    lab = cv2.imread(os.path.join(p2, item), 0)
    # lab = yimage.io.read_image(os.path.join(p2, item))
    print(set(lab.flatten()))
    if item in files1:
      id+=1

print(id)
print(len(files2))