import yimage
import numpy as np
import os
import tqdm
from tools.utils import parse_color_table

p_big = r'F:\0github\github_master\vai_data\val_gt'
p_src = r'F:\0github\github_master\vai_data\cut_data\label_val'
color_txt = r'F:\0github\github_master\vai_data\color_table_isprs.txt'
pred_merge = r'F:\0github\github_master\result_merge'
cut_size = 512
over_lap = 256
suffix = 'tif'

imgs = os.listdir(p_big)

for item in tqdm.tqdm(imgs):
    lab = yimage.io.read_image(os.path.join(p_big, item))
    h, w = lab.shape
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
            # save_data(img_s, lab_s,
            #           os.path.join(save_dir, 'image_train', '{}_{}_{}.{}'.format(img_name, ni, nj, suffix)),
            #           os.path.join(save_dir, 'label_train', '{}_{}_{}.{}'.format(lab_name, ni, nj, suffix)))
            cut_lab = yimage.io.read_image(
                os.path.join(p_src, '{}_{}_{}.{}'.format(item.split('.')[0], ni, nj, suffix)))
            pred[:, left - cut_size:left][down - cut_size:down, :] += cut_lab
            count_idx[:, left - cut_size:left][down - cut_size:down, :] += 1
            down = down + cut_size - over_lap
        down = cut_size
        left = left + cut_size - over_lap
    pred = pred/count_idx
    pred = pred.astype(np.uint8)
    pred = pred[pad_u:-pad_d, pad_l:-pad_r]
    yimage.io.write_image(os.path.join(pred_merge, item), pred + 1,
                          color_table=parse_color_table(color_txt))


