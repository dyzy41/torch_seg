import os
import tqdm
import numpy as np
from tools.metrics import get_acc_info
import ttach as tta

import tools.utils
from tools.utils import label_mapping
from networks.get_model import get_net
from collections import OrderedDict
import yimage
import tools.transform as tr
import torchvision.transforms as transforms
import torch
import torch.nn.functional as F
from tools.utils import read_image


# def tta_inference(inp, model, num_classes=8, scales=[1.0], flip=True):
#     b, _, h, w = inp.size()
#     preds = inp.new().resize_(b, num_classes, h, w).zero_().to(inp.device)
#     for scale in scales:
#         size = (int(scale * h), int(scale * w))
#         resized_img = F.interpolate(inp, size=size, mode='bilinear', align_corners=True, )
#         pred = model_inference(model, resized_img.to(inp.device), flip)
#         pred = F.interpolate(pred, size=(h, w), mode='bilinear', align_corners=True, )
#         preds += pred
#
#     return preds / (len(scales))


def tta_inference(inp, model):
    tta_transforms = tta.Compose(
        [
            tta.VerticalFlip(),
            tta.HorizontalFlip(),
            tta.Rotate90(angles=[0, 180]),
            # tta.Scale(scales=[1, 2, 4]),
            # tta.Multiply(factors=[0.8, 1, 1.25]),
        ]
    )
    with torch.no_grad():
        tta_model = tta.SegmentationTTAWrapper(model, tta_transforms, merge_mode='mean')
        preds = tta_model(inp)
        return preds


def inference(image, model):
    with torch.no_grad():
        output = model(image)
        return output


def load_model(model_path):
    model = get_net(param_dict['model_name'], param_dict['input_bands'], param_dict['num_class'],
                    param_dict['img_size'], param_dict['pretrained_model'])
    # model = torch.nn.DataParallel(model, device_ids=[0])
    state_dict = torch.load(model_path)['net']
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k[7:]
        new_state_dict[name] = v
    model.load_state_dict(state_dict)
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


if __name__ == '__main__':
    from tools.parse_config_yaml import parse_yaml

    # import pudb;pu.db
    param_dict = parse_yaml('config.yaml')
    print(param_dict)
    test_path = r'C:\Users\admin\Documents\0Ccode\cls3data\val'
    test_label = r'C:\Users\admin\Documents\0Ccode\cls3data\valannot'
    model_path = os.path.join(param_dict['model_dir'], 'valiou_best.pth')
    print('####################test model is {}'.format(model_path))

    if os.path.exists(param_dict['pred_path']) is False:
        os.mkdir(param_dict['pred_path'])

    model = load_model(model_path)

    test_imgs = os.listdir(test_path)
    for name in tqdm.tqdm(test_imgs):
        input_image = read_image(os.path.join(test_path, name)).astype(np.float32)
        input_image = img_transforms(input_image)
        input_image = input_image.unsqueeze(0).cuda()
        if param_dict['tta']:
            output = tta_inference(input_image, model)
        else:
            output = inference(input_image, model)
        pred_gray = tools.utils.out2pred(output, param_dict['num_class'], param_dict['thread'])
        yimage.io.write_image(os.path.join(param_dict['pred_path'], name), pred_gray[0], color_table=param_dict['color_table'])

    precision, recall, f1ccore, OA, IoU, mIOU = get_acc_info(param_dict['pred_path'], test_label, param_dict['num_class'] + 1 if param_dict['num_class'] == 1 else param_dict['num_class'],)
