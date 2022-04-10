import os
import cv2
# import yimage
import numpy as np
import tqdm

p2 = 'dataset/label_data/label_train'

files2 = os.listdir(p2)
id = 0
for item in tqdm.tqdm(files2):
    lab = cv2.imread(os.path.join(p2, item), 0)
    # lab = yimage.io.read_image(os.path.join(p2, item))
    lab = np.where(lab == 64, 1, lab)
    lab = np.where(lab == 128, 2, lab)
    lab = np.where(lab == 255, 3, lab)
    cv2.imwrite(os.path.join(p2, item), lab)
