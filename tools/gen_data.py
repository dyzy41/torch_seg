import os
import numpy as np
import cv2
import glob
import SimpleITK as sitk
import tqdm


def mha2jpg(mhaPath, src_path, out_name, flag, windowsCenter=512, windowsSize=1024):
    """
    The function can output a group of jpg files by a specified mha file.
    Args:
        mhaPath:mha file path.
        outfolder:The folder that the jpg files are saved.
        windowsCenter:the CT windows center.
        windowsSize:the CT windows size.
    Return:void

    """
    image = sitk.ReadImage(mhaPath)
    img_data = sitk.GetArrayFromImage(image)
    channel = img_data.shape[0]

    low = windowsCenter - windowsSize / 2
    high = windowsCenter + windowsSize / 2

    for s in range(channel):
        slicer = img_data[s, :, :]
        slicer[slicer < low] = low
        slicer[slicer > high] = high
        slicer = slicer - low
        img = cv2.normalize(slicer, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
        cv2.imwrite(os.path.join(label_save_path, '{}_{}_{}.png'.format(out_name, s, flag)), img)


def convert_from_dicom_to_jpg(img, low_window, high_window, save_path):
    lungwin = np.array([low_window * 1., high_window * 1.])
    newimg = (img - lungwin[0]) / (lungwin[1] - lungwin[0])  # 归一化
    newimg = (newimg * 255).astype('uint8')  # 将像素值扩展到[0,255]
    cv2.imwrite(save_path, newimg, [int(cv2.IMWRITE_JPEG_QUALITY), 100])


def parse_dcm(dcm_files, person_name, flag):
    for id, item in tqdm.tqdm(enumerate(dcm_files)):
        ds_array = sitk.ReadImage(item)  # 读取dicom文件的相关信息
        img_array = sitk.GetArrayFromImage(ds_array)  # 获取array
        # SimpleITK读取的图像数据的坐标顺序为zyx，即从多少张切片到单张切片的宽和高，此处我们读取单张，因此img_array的shape
        # 类似于 （1，height，width）的形式
        shape = img_array.shape
        img_array = np.reshape(img_array, (shape[1], shape[2]))  # 获取array中的height和width
        high = np.max(img_array)
        low = np.min(img_array)
        convert_from_dicom_to_jpg(img_array, low, high, os.path.join(image_save_path, '{}_{}_{}.png'.format(person_name,
                                                                                                         id, flag)))  # 调用函数，转换成jpg文件并保存到对应的路


def gen_img(path, flag):
    sub_dirs = os.listdir(path)
    for item in sub_dirs:
        dcm_files = os.listdir(os.path.join(path, item))
        dcm_files = [i for i in dcm_files if i.endswith('DCM')]
        dcm_files.sort(key=lambda x: int(x.split('.')[0]))
        dcm_files = [os.path.join(path, item, i) for i in dcm_files]

        parse_dcm(dcm_files, item, flag)
        mha_file = os.path.join(path, item, '{}.mha'.format(item))
        mha2jpg(mha_file, path, item, flag)


if __name__ == '__main__':
    health = './dataset/label_data/HEALTH'
    ill = './dataset/label_data/ILL'
    image_save_path = './dataset/label_data/image_train'
    label_save_path = './dataset/label_data/label_train'
    # os.mkdir(image_save_path)
    # os.mkdir(label_save_path)
    gen_img(health, flag='health')
    gen_img(ill, flag='ill')
