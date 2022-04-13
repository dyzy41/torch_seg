import os

import yimage
import tqdm
import numpy as np

src_path = r'F:\0DL_datasets\WHU_building\whub'


def resave_lab(path):
    files = os.listdir(path)
    for item in tqdm.tqdm(files):
        cur_lab = yimage.io.read_image(os.path.join(path, item))
        cur_lab = np.where(cur_lab > 0, 0, 1)
        yimage.io.write_image(os.path.join(path, item), cur_lab, color_table=ct)


if __name__ == '__main__':
    ct = [(0,0,0), (255,255,255)]
    resave_lab(os.path.join(src_path, 'train/label'))
    resave_lab(os.path.join(src_path, 'test/label'))
    resave_lab(os.path.join(src_path, 'val/label'))
