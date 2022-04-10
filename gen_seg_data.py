import os
import cv2
import tqdm
from os.path import join as osp
import numpy as np
from collections import OrderedDict
import SimpleITK as sitk
from tools.dcm2jpg import convert_from_dicom_to_jpg
import torchvision.transforms as transforms
import tools.transform as tr
import torch
from networks.get_model import get_net


def load_model(model_path):
    model = get_net(param_dict['model_name'], param_dict['input_bands'], param_dict['num_class'],
                    param_dict['img_size'], param_dict['pretrained_model'])
    # model = torch.nn.DataParallel(model, device_ids=[0])
    state_dict = torch.load(model_path)
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k[7:]
        new_state_dict[name] = v
    model.load_state_dict(state_dict)
    if param_dict['use_gpu']:
        model.cuda()
    model.eval()
    return model


def img_transforms(img):
    img = np.array(img).astype(np.float32)
    sample = {'image': img}
    transform = transforms.Compose([
        tr.FixedResize(param_dict['img_size']),
        tr.Normalize(mean=param_dict['mean'], std=param_dict['std']),
        tr.ToTensor()])
    sample = transform(sample)
    return sample['image']


def model_inference(model, image):
    with torch.no_grad():
        output = model(image)
        return output


def dcm2png(dcm_image_path, output_jpg_path):
    ds_array = sitk.ReadImage(dcm_image_path)  # 读取dicom文件的相关信息
    img_array = sitk.GetArrayFromImage(ds_array)  # 获取array
    # SimpleITK读取的图像数据的坐标顺序为zyx，即从多少张切片到单张切片的宽和高，此处我们读取单张，因此img_array的shape
    # 类似于 （1，height，width）的形式
    shape = img_array.shape
    img_array = np.reshape(img_array, (shape[1], shape[2]))  # 获取array中的height和width
    high = np.max(img_array)
    low = np.min(img_array)
    img3, img = convert_from_dicom_to_jpg(img_array, low, high, output_jpg_path)  # 调用函数，转换成jpg文件并保存到对应的路径
    return img3, img


def seg_dir(path, model, flag):
    person_files = os.listdir(path)
    if os.path.exists(osp(save_path, flag)) is False:
        os.mkdir(osp(save_path, flag))
    for item in tqdm.tqdm(person_files):
        if os.path.exists(osp(save_path, flag, item)) is False:
            os.mkdir(osp(save_path, flag, item))
        img_files = os.listdir(osp(path, item))
        img_files = [i for i in img_files if i.endswith('DCM')]
        # img_files = [osp(path, item, i) for i in img_files]
        for imgF in img_files:

            img3, img = dcm2png(osp(path, item, imgF), osp(save_path, flag, item, imgF.replace('DCM', 'png')))
            img3 = img3.astype(np.float32)
            scale_image = img_transforms(img3)
            scale_image = scale_image.unsqueeze(0).cuda()
            outputs = model_inference(model, scale_image)
            pred = torch.argmax(outputs, 1).detach().cpu().numpy()[0]
            truth = pred == 0
            img[truth] = 0
            left_img = img[:, :256]
            right_img = img[:, 256:]
            cv2.imwrite(osp(save_path, flag, item, imgF.replace('.DCM', '_0.png')), left_img)
            cv2.imwrite(osp(save_path, flag, item, imgF.replace('.DCM', '_1.png')), right_img)


if __name__ == '__main__':
    health_path = '../dataset/label_data/HEALTH'
    ill_path = '../dataset/label_data/ILL'
    save_path = '../dataset/seg_data'

    from tools.parse_config_yaml import parse_yaml

    # import pudb;pu.db
    param_dict = parse_yaml('config.yaml')
    print(param_dict)

    model_path = osp(param_dict['model_dir'], 'valiou_best.pth')
    print('####################test model is {}'.format(model_path))

    if os.path.exists(param_dict['save_path']) is False:
        os.mkdir(param_dict['save_path'])

    model = load_model(model_path)
    seg_dir(health_path, model, 'HEALTH')
    seg_dir(ill_path, model, 'ILL')
