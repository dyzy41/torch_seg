from __future__ import print_function, division
import os
import torch
import numpy as np
from torch.utils.data import Dataset
from tools import utils


class IsprsSegmentation(Dataset):
    """
    PascalVoc dataset
    """

    def __init__(self,
                 txt_path=None,
                 transform=None
                 ):
        self.list_sample = open(txt_path, 'r').readlines()
        self.num_sample = len(self.list_sample)
        assert self.num_sample > 0
        print('# samples: {}'.format(self.num_sample))
        self.transform = transform

    def __len__(self):
        return self.num_sample
        # return 256

    def __getitem__(self, index):
        _img, _target, img_path, gt_path = self._make_img_gt_point_pair(index)
        aug = self.transform(image=_img, mask=_target)
        sample = {'image': aug['image'], 'gt': aug['mask']}
        # if self.transform is not None:
        #     sample = self.transform(sample)
        sample['img_path'] = img_path
        sample['gt_path'] = gt_path
        return sample

    def _make_img_gt_point_pair(self, index):
        file = self.list_sample[index].strip()
        img_path = file.split('  ')[0]
        gt_path = file.split('  ')[1]
        _img = utils.read_image(os.path.join(img_path))
        _target = utils.read_image(os.path.join(gt_path), 'gt').astype(np.int32)
        return _img, _target, img_path, gt_path
