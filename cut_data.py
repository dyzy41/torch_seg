import yimage
import numpy as np
import os
import tqdm
import cv2


def read_data(p_dict):
    p_img = p_dict[0]
    p_lab = p_dict[1]
    img = yimage.io.read_image(p_img)
    lab = yimage.io.read_image(p_lab)
    # print(set(img.flatten()))
    return img, lab


def save_data(img_s, lab_s, p_img, p_lab):
    yimage.io.write_image(p_img, img_s)
    yimage.io.write_image(p_lab, lab_s)
    # cv2.imwrite(p_lab, lab_s)


def gen_dict():
    imgs = os.listdir(big_img_path)
    labs = os.listdir(big_gt_path)
    imgs = [os.path.join(big_img_path, i) for i in imgs]
    labs = [os.path.join(big_gt_path, i) for i in labs]
    path_list = []
    for i in range(len(imgs)):
        path_list.append([imgs[i], labs[i]])
    return path_list


def cut_data(cut_size, over_lap, save_dir):
    path_list = gen_dict()
    if os.path.exists(save_dir) is False:
        os.mkdir(save_dir)
        os.mkdir(os.path.join(save_dir, 'image_train'))
        os.mkdir(os.path.join(save_dir, 'label_train'))

    for i in tqdm.tqdm(range(len(path_list))):
        img_name = path_list[i][0].replace('\\', '/').split('/')[-1].replace('.' + suffix, '')
        lab_name = path_list[i][1].replace('\\', '/').split('/')[-1].replace('.' + suffix, '')
        img, lab = read_data(path_list[i])
        h, w = lab.shape
        down, left = cut_size, cut_size
        h_new = ((h - cut_size) // (cut_size - over_lap) + 1) * (cut_size - over_lap) + cut_size
        h_pad = h_new - h
        w_new = ((w - cut_size) // (cut_size - over_lap) + 1) * (cut_size - over_lap) + cut_size
        w_pad = w_new - w

        # img_ = cv2.copyMakeBorder(img, h_pad, 0, w_pad, 0, cv2.BORDER_REFLECT)
        # lab_ = cv2.copyMakeBorder(lab, h_pad, 0, w_pad, 0, cv2.BORDER_REFLECT)
        pad_u = h_pad//2
        pad_d = h_pad - pad_u
        pad_l = w_pad//2
        pad_r = w_pad-pad_l

        # lab = np.pad(lab, ((0, h_pad), (0, w_pad)), 'reflect')
        # img = np.pad(img, ((0, h_pad), (0, w_pad), (0, 0)), 'reflect')
        lab = np.pad(lab, ((pad_u, pad_d), (pad_l, pad_r)), 'reflect')
        img = np.pad(img, ((pad_u, pad_d), (pad_l, pad_r), (0, 0)), 'reflect')
        # yimage.io.write_image('ok.tif', img_)


        ni = 0
        while left <= w_new:
            slice_img = img[:, left - cut_size:left, :]
            slice_lab = lab[:, left - cut_size:left]
            ni += 1
            nj = 0
            while down <= h_new:
                img_s = slice_img[down - cut_size:down, :, :]
                lab_s = slice_lab[down - cut_size:down, :]

                nj += 1
                save_data(img_s, lab_s, os.path.join(save_dir, 'image_train', '{}_{}_{}.{}'.format(img_name, ni, nj, suffix)),
                          os.path.join(save_dir, 'label_train', '{}_{}_{}.{}'.format(lab_name, ni, nj, suffix)))
                down = down + cut_size - over_lap
            down = cut_size
            left = left + cut_size - over_lap

    print('finished data cutting')


if __name__ == '__main__':
    root_path = './data'
    big_img_path = r'../../vai_data/train_img/train'
    big_gt_path = r'../../vai_data/dsm/train'
    cut_size = 512
    over_lap = 256
    save_dir = '../../vai_data/data_slice2'
    suffix = 'tif'
    cut_data(cut_size, over_lap, save_dir)
