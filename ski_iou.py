import numpy as np
import os
from tools.utils import read_image

pred_path = r'U:\private\dongsj\CUG_seg\result2'
gt_path = r'U:\private\dongsj\CUG_seg\CHN6-CUG\val\gt'

img_size = (512, 512)
classes = np.array([0, 1]).astype('uint8')
files = os.listdir(pred_path)

res = []
for cls in classes:
    D = np.zeros([len(files), img_size[0], img_size[1], 2]).astype(bool)
    for i, file in enumerate(files):
        img1 = read_image(os.path.join(pred_path, file))
        img2 = read_image(os.path.join(gt_path, file))
        img2 = np.where(img2 > 0, 1, 0)

        D[i, :, :, 0] = img1 == cls
        D[i, :, :, 1] = img2 == cls
    res.append(np.sum(D[..., 0] & D[..., 1]) / np.sum(D[..., 0] | D[..., 1]))

for i, cls in enumerate(classes):
    print('class ' + str(cls) + ' :' + str(res[i]))


# class 0 :0.970986573337791
# class 1 :0.6284760231166898