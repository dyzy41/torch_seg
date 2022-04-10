import os
import yimage

import numpy as np
import tqdm

def label_mapping(label_im):
    # colorize = np.zeros([2, 3], dtype=np.int64)
    colorize = np.array([[0, 0, 0],
                [255, 0, 0],
                [0, 255, 0],
                [0, 0, 255],
                [0, 0, 255],
                [128, 128, 0],
                [192, 128, 128],
                [64, 64, 128],
                [64, 0, 128],
                [64, 64, 0],
                [0, 128, 192],
                [255, 255, 255]
                ])
    label = colorize[label_im, :].reshape([label_im.shape[0], label_im.shape[1], 3])
    return label


GT_ICM = './dataset/人胚胎数据集/BlastsOnline/GT_ICM'
GT_TE = './dataset/人胚胎数据集/BlastsOnline/GT_TE'
GT_ZP = './dataset/人胚胎数据集/BlastsOnline/GT_ZP'
Images = './dataset/人胚胎数据集/BlastsOnline/Images'
save = './dataset/process_data/labels'
save_vis = './dataset/process_data/labels_vis'

names = os.listdir(Images)

for item in tqdm.tqdm(names):
    icm = yimage.io.read_image(os.path.join(GT_ICM, item.replace('.BMP', ' ICM_Mask.bmp')))
    te = yimage.io.read_image(os.path.join(GT_TE, item.replace('.BMP', ' TE_Mask.bmp')))
    zp = yimage.io.read_image(os.path.join(GT_ZP, item.replace('.BMP', ' ZP_Mask.bmp')))
    size = icm.shape
    label = np.zeros(size)


    label = np.where(icm > 0, 1, label)
    label = np.where(te > 0, 3, label)
    label = np.where(zp > 0, 2, label)
    label = label.astype('uint8')
    concat_vis = label_mapping(label)
    yimage.io.write_image(os.path.join(save, item), label)
    yimage.io.write_image(os.path.join(save_vis, item), concat_vis)



