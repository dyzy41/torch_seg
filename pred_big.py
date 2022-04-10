# -*- coding:utf-8 -*-
import cv2
import csv
import tqdm
from collections import OrderedDict
from PIL import Image
import numpy as np
from torch.autograd import Variable
import os
import tools.transform as tr

np.seterr(divide='ignore', invalid='ignore')
import torch
import torchvision.transforms as transforms
import torchvision.models as models
import shutil
from networks.get_net import get_net
from tools.cal_iou import evaluate
from config import *


def img_transforms(img):
    img = np.array(img).astype(np.float32)
    sample = {'image': img}
    transform = transforms.Compose([
        tr.Normalize(mean=mean, std=std),
        tr.ToTensor()])
    sample = transform(sample)
    return sample['image']


def find_new_file(dir):
    if os.path.exists(dir) is False:
        os.mkdir(dir)
        dir = dir

    file_lists = os.listdir(dir)
    file_lists.sort(key=lambda fn: os.path.getmtime(dir + fn)
    if not os.path.isdir(dir + fn) else 0)
    if len(file_lists) != 0:
        file = os.path.join(dir, file_lists[-1])
        return file
    else:
        return None



def test_my(input_bands, model_name, model_dir, img_size, num_class, image_path, label_path):
    net = get_net(model_name, input_bands, num_class, img_size)
    imgs = os.listdir(image_path)
    
    if os.path.exists(output):
        shutil.rmtree(output)
        os.mkdir(output)
    else:
        os.mkdir(output)
    if os.path.exists(output_gray):
        shutil.rmtree(output_gray)
        os.mkdir(output_gray)
    else:
        os.mkdir(output_gray)
    state_dict = torch.load(find_new_file(model_dir))
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k[7:]
        new_state_dict[name] = v
    net.load_state_dict(new_state_dict)
    if use_gpu:
        net.cuda()
    net.eval()
    for i in tqdm.tqdm(range(len(imgs))):
        img_path = output + imgs[i][0:-5] + '/'
        os.mkdir(img_path)
        pred_image(net, os.path.join(image_path, imgs[i]), output, output_gray, img_path)

    iou, acc, recall, precision = evaluate(label_path, output_gray, num_class)
    return iou, acc, recall, precision


if __name__ == '__main__':
    x, y, recall, precision = test_my(input_bands, model_name, model_dir, img_size, num_class, test_path, test_gt)
    print([x, y, recall, precision])
