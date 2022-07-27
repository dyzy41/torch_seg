import yimage
import numpy as np
import os
import tqdm
import sys
from parse_config_yaml import parse_yaml


def process():
    imgs = os.listdir(p_bigimg)

    for item in tqdm.tqdm(imgs):
        bigimg = yimage.io.read_image(os.path.join(p_bigimg, item))
        h, w, _ = bigimg.shape
        count_idx = np.zeros((h, w))
        down, left = cut_size, cut_size
        h_new = ((h - cut_size) // (cut_size - over_lap) + 1) * (cut_size - over_lap) + cut_size
        h_pad = h_new - h
        w_new = ((w - cut_size) // (cut_size - over_lap) + 1) * (cut_size - over_lap) + cut_size
        w_pad = w_new - w

        pad_u = h_pad // 2
        pad_d = h_pad - pad_u
        pad_l = w_pad // 2
        pad_r = w_pad - pad_l

        count_idx = np.pad(count_idx, ((pad_u, pad_d), (pad_l, pad_r)), 'reflect')
        pred = count_idx.copy()

        ni = 0
        while left <= w_new:
            slice_pred = pred[:, left - cut_size:left]

            ni += 1
            nj = 0
            while down <= h_new:
                lab_s = slice_pred[down - cut_size:down, :]
                nj += 1
                cut_lab = yimage.io.read_image(
                    os.path.join(p_predslice, '{}_{}_{}.{}'.format(item.split('.')[0], ni, nj, suffix)))
                pred[:, left - cut_size:left][down - cut_size:down, :] += cut_lab
                count_idx[:, left - cut_size:left][down - cut_size:down, :] += 1
                down = down + cut_size - over_lap
            down = cut_size
            left = left + cut_size - over_lap
        pred = pred / count_idx
        pred = pred.astype(np.uint8)
        pred = pred[pad_u:-pad_d, pad_l:-pad_r]
        yimage.io.write_image(os.path.join(p_predbig, item), pred,
                              color_table=ct)

if __name__ == '__main__':
    if len(sys.argv) == 1:
        yaml_file = 'config.yaml'
    else:
        yaml_file = sys.argv[1]
    param_dict = parse_yaml(yaml_file)
    p_bigimg = ''
    p_predslice = param_dict['pred_path']
    ct = [(0, 0, 0), (255, 255, 255)]
    p_predbig = ''
    cut_size = 224
    over_lap = 32
    suffix = 'tif'
