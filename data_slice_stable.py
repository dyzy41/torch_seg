import cv2
import os
from config import *
import tqdm
from multiprocessing import Pool

def cut_img(img, name, overlap, save_path):
    shape = img.shape
    xratio = shape[0] // (img_size - overlap)
    yratio = shape[1] // (img_size - overlap)
    new_size = ((xratio + 2) * (img_size - overlap) + overlap, (yratio + 2) * (img_size - overlap) + overlap)
    new_img = cv2.resize(img, new_size)

    for i in range(xratio + 1):
        for j in range(yratio + 1):
            if i == 0:
                newx = 0
            else:
                newx = i * (img_size - overlap)
            if j == 0:
                newy = 0
            else:
                newy = j * (img_size - overlap)
            current_img = new_img[newx:newx + img_size, newy:newy + img_size, :]
            cur_name = name.split('.')[0] + '_{}_{}'.format(i, j) + '.' + name.split('.')[1]
            # print(cur_name)
            cv2.imwrite(os.path.join(save_path, cur_name), current_img)


def cut_lab(lab, name, overlap, save_path):
    shape = lab.shape
    xratio = shape[0] // (img_size - overlap)
    yratio = shape[1] // (img_size - overlap)
    new_size = ((xratio + 2) * (img_size - overlap) + overlap, (yratio + 2) * (img_size - overlap) + overlap)
    new_lab = cv2.resize(lab, new_size)
    assert len(set(new_lab.flatten())) != len(set(lab.flatten())), 'resize wrong!!!'

    for i in range(xratio + 1):
        for j in range(yratio + 1):
            if i == 0:
                newx = 0
            else:
                newx = i * (img_size - overlap)
            if j == 0:
                newy = 0
            else:
                newy = j * (img_size - overlap)
            current_lab = new_lab[newx:newx + img_size, newy:newy + img_size]
            cur_name = name.split('.')[0] + '_{}_{}'.format(i, j) + '.' + name.split('.')[1]
            cv2.imwrite(os.path.join(save_path, cur_name), current_lab)


def do(name, pimg, plab, target_path, state):
    img = cv2.imread(os.path.join(pimg, name))
    lab = cv2.imread(os.path.join(plab, name))
    cut_img(img, name, overlap, os.path.join(target_path, 'image_{}'.format(state)))
    cut_lab(lab, name, overlap, os.path.join(target_path, 'label_{}'.format(state)))


def slice_data(pimg, plab, state):
    src_img = os.listdir(pimg)
    target_path = data_dir
    if os.path.exists(target_path) is False:
        os.mkdir(target_path)
        os.mkdir(os.path.join(target_path, 'image_{}'.format(state)))
        os.mkdir(os.path.join(target_path, 'label_{}'.format(state)))
    if state == 'val':
        os.mkdir(os.path.join(target_path, 'image_{}'.format(state)))
        os.mkdir(os.path.join(target_path, 'label_{}'.format(state)))

    pimg_ = [pimg for i in range(len(src_img))]
    plab_ = [plab for i in range(len(src_img))]
    target_path_ = [target_path for i in range(len(src_img))]
    state_ = [state for i in range(len(src_img))]
    p = Pool(64)
    params = list(zip(src_img, pimg_, plab_, target_path_, state_))
    p.starmap(do, params)
    p.close()
    p.join()


if __name__ == '__main__':
    print('slice train dataset')
    slice_data(train_path, train_gt, 'train')
    print('slice val dataset')
    slice_data(val_path, val_gt, 'val')
    print('finished')

