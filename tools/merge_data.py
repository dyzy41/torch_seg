import yimage
import numpy as np
import os
import tqdm

p_big = r'C:\Users\admin\Documents\0Ccode\code220521_zqseg\dataset\zhaoq\big_img'
p_src = r'C:\Users\admin\Documents\0Ccode\code220521_zqseg\0401_files\Res_UNet_50_v1\test_result'
ct = [(0,0,0), (255,255,255)]
pred_merge = r'C:\Users\admin\Documents\0Ccode\code220521_zqseg\dataset\zhaoq\big_pred'
cut_size = 256
over_lap = 64
suffix = 'tif'

imgs = os.listdir(p_big)

for item in tqdm.tqdm(imgs):
    bigimg = yimage.io.read_image(os.path.join(p_big, item))
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
                os.path.join(p_src, '{}_{}_{}.{}'.format(item.split('.')[0], ni, nj, suffix)))
            pred[:, left - cut_size:left][down - cut_size:down, :] += cut_lab
            count_idx[:, left - cut_size:left][down - cut_size:down, :] += 1
            down = down + cut_size - over_lap
        down = cut_size
        left = left + cut_size - over_lap
    pred = pred/count_idx
    pred = pred.astype(np.uint8)
    pred = pred[pad_u:-pad_d, pad_l:-pad_r]
    yimage.io.write_image(os.path.join(pred_merge, item), pred,
                          color_table=ct)


