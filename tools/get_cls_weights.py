import os
import yimage
import numpy as np
from collections import Counter
import tqdm


def count_data(path):
    files = os.listdir(path)
    info = []
    for item in tqdm.tqdm(files):
        img = yimage.io.read_image(os.path.join(path, item))
        info += list(img.flatten())
    return Counter(info), len(info)


def count_dir(path):
    d2, num2 = count_data(path)
    c_dict2 = dict(d2)
    c_dict2 = sorted(c_dict2.items(), key=lambda x: x[0])
    c_dict2 = [(int(i[0]), i[1] * 100.00 / num2) for i in c_dict2]
    return c_dict2


if __name__ == '__main__':
    p = r'D:\zd_seg_v2\zd_data\train_label'
    p2 = r'D:\zd_seg_v2\zd_data\val_label'
    p3 = r'D:\zd_seg_v2\zd_data\test_label'
    # d1 = count_dir(p)
    # d2 = count_dir(p2)
    # d3 = count_dir(p3)
    p4 = '/home/dsj/torch_seg_debug/dataset/voc/cut_data/label_train'
    d4 = count_dir(p4)
    kk = [i[1] for i in d4]

    print(kk)

# 14.036847451623427, 13.822370076515773, 9.221586582639937, 17.25225099200799, 24.1940533751087, 10.048654178872225, 3.396645422767445, 7.242551090015006, 0.7850408304494938
# [14.479022654316958, 14.023344092848285, 22.523172390570394, 6.8471790391199345, 4.00002707184943, 22.609023529516474, 1.3396374893937686, 13.663987542499095, 0.5146061898856604]

# voc, 78.08040418536484, 0.5634292146818277, 0.253603248059279, 0.6804753364748943, 0.4585480878249174, 0.5731837347615351, 1.389097455355706, 1.166330528062846, 2.0011722448147986, 0.9148914299181471, 0.8221193805331837, 1.182792655716964, 1.5196898775005767, 0.8083318624552116, 0.8759743153250917, 4.307584176552979, 0.4973709153588842, 0.6542078784925602, 1.311876615870666, 1.280313891886519, 0.658602964988575