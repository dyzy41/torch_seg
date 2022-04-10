import cv2
import numpy as np
from utils.utils import label_mapping
from config import *
from networks.get_net import get_net
from collections import OrderedDict
import yimage
import utils.transform as tr
import torchvision.transforms as transforms
import torch
import torch.nn.functional as F
from PIL import Image

def img_transforms(img):
    img = np.array(img).astype(np.float32)
    sample = {'image': img}
    transform = transforms.Compose([
        tr.FixedResize(img_size),
        tr.Normalize(mean=mean, std=std),
        tr.ToTensor()])
    sample = transform(sample)
    return sample['image']

def load_model(model_path):
    model = get_net(model_name, input_bands, num_class, img_size)
    state_dict = torch.load(model_path)
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k[7:]
        new_state_dict[name] = v
    model.load_state_dict(new_state_dict)
    if use_gpu:
        model.cuda()
    model.eval()
    return model

lab_path = r'Y:\private\dongsj\0sjcode\code0906_vaiseg\vai_data\val_gt\top_mosaic_09cm_area1.tif'
img_path = r'Y:\private\dongsj\0sjcode\code0906_vaiseg\vai_data\cut_data\image_val\top_mosaic_09cm_area1_1_5.tif'
model_path = r'Y:\private\dongsj\0sjcode\code0906_vaiseg\0913_files\DeepLabV3Plus_3\pth_DeepLabV3Plus\101.pth'
# lab_path = '/nfs/project/netdisk/192.168.0.31/d/private/dongsj/0sjcode/code0906_vaiseg/vai_data/train_gt/top_mosaic_09cm_area1.tif'
# img_path = '/nfs/project/netdisk/192.168.0.31/d/private/dongsj/0sjcode/code0906_vaiseg/vai_data/train_img/val/top_mosaic_09cm_area1.tif'
# model_path = '/nfs/project/netdisk/192.168.0.31/d/private/dongsj/0sjcode/code0906_vaiseg/0910_test_files/DeepLabV3Plus_3/pth_DeepLabV3Plus/491.pth'

model = load_model(model_path)
# image = yimage.io.read_image(img_path)
image = np.asarray(Image.open(img_path).convert('RGB')).astype(np.float32)

scale_image = img_transforms(image)
scale_image = scale_image.unsqueeze(0).cuda()

model.eval()
with torch.no_grad():
    output = model(scale_image)
    pred_gray = torch.argmax(output, 1)
    pred_gray = pred_gray[0].cpu().data.numpy().astype(np.int32)
    pred_vis = label_mapping(pred_gray)
    cv2.imwrite('test_vis22.tif', pred_vis)