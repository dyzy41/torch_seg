import os
import tqdm
import cv2
import numpy as np
from tools.metrics import get_acc_info

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


def tta_inference(inp, model, num_classes=8, scales=[1.0], flip=True):
    b, _, h, w = inp.size()
    preds = inp.new().resize_(b, num_classes, h, w).zero_().to(inp.device)
    for scale in scales:
        size = (int(scale * h), int(scale * w))
        resized_img = F.interpolate(inp, size=size, mode='bilinear', align_corners=True, )
        pred = model_inference(model, resized_img.to(inp.device), flip)
        pred = F.interpolate(pred, size=(h, w), mode='bilinear', align_corners=True, )
        preds += pred

    return preds / (len(scales))


def model_inference(model, image, flip=True):
    if param_dict['extra_loss']:
        with torch.no_grad():
            output = model(image)[0]
            if flip:
                fimg = image.flip(2)
                output += model(fimg)[0].flip(2)
                fimg = image.flip(3)
                output += model(fimg)[0].flip(3)
                return output / 3
            return output
    else:
        with torch.no_grad():
            output = model(image)
            if flip:
                fimg = image.flip(2)
                output += model(fimg).flip(2)
                fimg = image.flip(3)
                output += model(fimg).flip(3)
                return output / 3
            return output


def pred_img(model, image):
    with torch.no_grad():
        output = model(image)
    return output


def slide_pred(model, image_path, num_classes=6, crop_size=512, overlap=256, scales=[1.0], flip=True, file_name=None):
    # scale_image = yimage.io.read_image(image_path)
    scale_image = read_image(image_path).astype(np.float32)
    # scale_image = np.asarray(Image.open(image_path).convert('RGB')).astype(np.float32)
    scale_image = img_transforms(scale_image)
    scale_image = scale_image.unsqueeze(0).cuda()

    N, C, H_, W_ = scale_image.shape
    print(f"Height: {H_} Width: {W_}")

    full_probs = torch.zeros((N, num_classes, H_, W_), device=scale_image.device)  #
    count_predictions = torch.zeros((N, num_classes, H_, W_), device=scale_image.device)  #

    h_overlap_length = overlap
    w_overlap_length = overlap

    h = 0
    slide_finish = False
    slice_id = 0
    while not slide_finish:

        if h + crop_size <= H_:
            # print(f"h: {h}")
            # set row flag
            slide_row = True
            # initial row start
            w = 0
            while slide_row:
                if w + crop_size <= W_:
                    # print(f" h={h} w={w} -> h'={h + crop_size} w'={w + crop_size}")
                    patch_image = scale_image[:, :, h:h + crop_size, w:w + crop_size]
                    #
                    patch_pred_image = tta_inference(patch_image, model, num_classes=num_classes, scales=scales,
                                                     flip=flip)
                    patch_gray = torch.argmax(patch_pred_image, 1)
                    patch_gray = patch_gray[0].cpu().data.numpy().astype(np.int32)
                    yimage.io.write_image(
                        os.path.join(param_dict['save_path'], 'slice', '{}_{}.tif'.format(file_name[:-4], slice_id)),
                        patch_gray + 1,
                        color_table=tools.utils.parse_color_table(param_dict['color_txt']))
                    slice_id += 1
                    # patch_pred_image = pred_img(model, patch_image)
                    count_predictions[:, :, h:h + crop_size, w:w + crop_size] += 1
                    full_probs[:, :, h:h + crop_size, w:w + crop_size] += patch_pred_image

                else:
                    # print(f" h={h} w={W_ - crop_size} -> h'={h + crop_size} w'={W_}")
                    patch_image = scale_image[:, :, h:h + crop_size, W_ - crop_size:W_]
                    #
                    patch_pred_image = tta_inference(patch_image, model, num_classes=num_classes, scales=scales,
                                                     flip=flip)
                    patch_gray = torch.argmax(patch_pred_image, 1)
                    patch_gray = patch_gray[0].cpu().data.numpy().astype(np.int32)
                    yimage.io.write_image(
                        os.path.join(param_dict['save_path'], 'slice', '{}_{}.tif'.format(file_name[:-4], slice_id)),
                        patch_gray + 1,
                        color_table=tools.utils.parse_color_table(param_dict['color_txt']))
                    slice_id += 1
                    # patch_pred_image = pred_img(model, patch_image)
                    count_predictions[:, :, h:h + crop_size, W_ - crop_size:W_] += 1
                    full_probs[:, :, h:h + crop_size, W_ - crop_size:W_] += patch_pred_image
                    slide_row = False

                w += w_overlap_length

        else:
            # print(f"h: {h}")
            # set last row flag
            slide_last_row = True
            # initial row start
            w = 0
            while slide_last_row:
                if w + crop_size <= W_:
                    # print(f"h={H_ - crop_size} w={w} -> h'={H_} w'={w + crop_size}")
                    patch_image = scale_image[:, :, H_ - crop_size:H_, w:w + crop_size]
                    #
                    patch_pred_image = tta_inference(patch_image, model, num_classes=num_classes, scales=scales,
                                                     flip=flip)
                    patch_gray = torch.argmax(patch_pred_image, 1)
                    patch_gray = patch_gray[0].cpu().data.numpy().astype(np.int32)
                    yimage.io.write_image(
                        os.path.join(param_dict['save_path'], 'slice', '{}_{}.tif'.format(file_name[:-4], slice_id)),
                        patch_gray + 1,
                        color_table=tools.utils.parse_color_table(param_dict['color_txt']))
                    slice_id += 1
                    count_predictions[:, :, H_ - crop_size:H_, w:w + crop_size] += 1
                    full_probs[:, :, H_ - crop_size:H_, w:w + crop_size] += patch_pred_image

                else:
                    # print(f"h={H_ - crop_size} w={W_ - crop_size} -> h'={H_} w'={W_}")
                    patch_image = scale_image[:, :, H_ - crop_size:H_, W_ - crop_size:W_]
                    #
                    patch_pred_image = tta_inference(patch_image, model, num_classes=num_classes, scales=scales,
                                                     flip=flip)
                    patch_gray = torch.argmax(patch_pred_image, 1)
                    patch_gray = patch_gray[0].cpu().data.numpy().astype(np.int32)
                    yimage.io.write_image(
                        os.path.join(param_dict['save_path'], 'slice', '{}_{}.tif'.format(file_name[:-4], slice_id)),
                        patch_gray + 1,
                        color_table=tools.utils.parse_color_table(param_dict['color_txt']))
                    slice_id += 1
                    count_predictions[:, :, H_ - crop_size:H_, W_ - crop_size:W_] += 1
                    full_probs[:, :, H_ - crop_size:H_, W_ - crop_size:W_] += patch_pred_image

                    slide_last_row = False
                    slide_finish = True

                w += w_overlap_length

        h += h_overlap_length

    full_probs /= count_predictions

    return full_probs


def load_model(model_path):
    model = get_net(param_dict['model_name'], param_dict['input_bands'], param_dict['num_class'],
                    param_dict['img_size'], param_dict['pretrained_model'])
    state_dict = torch.load(model_path)  # 加载断点
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
    test_path = r'D:\Yubo\torch_learn\code220314_whuseg\whub\test\image'
    test_label = r'D:\Yubo\torch_learn\code220314_whuseg\whub\test\label'
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
        output = tta_inference(input_image, model, num_classes=param_dict['num_class'], scales=[1.0], flip=True)
        pred_gray = tools.utils.out2pred(output, param_dict['num_class'], param_dict['thread'])
        yimage.io.write_image(os.path.join(param_dict['save_path'], name), pred_gray[0], color_table=param_dict['color_table'])

    precision, recall, f1ccore, OA, IoU, mIOU = get_acc_info(param_dict['save_path'], test_label, param_dict['num_class'] + 1 if param_dict['num_class'] == 1 else param_dict['num_class'],)
