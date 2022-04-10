import os
import numpy as np
from matplotlib import pyplot as plt
from tools.parse_config_yaml import parse_yaml
import tqdm


def parse_txt(txt_path):
    cur_info = open(txt_path, 'r').readlines()
    prec_value = float(cur_info[4].strip()[1:-1].replace('  ', ' ').split()[1])
    rec_value = float(cur_info[6].strip()[1:-1].replace('  ', ' ').split()[1])
    f1_value = float(cur_info[8].strip()[1:-1].replace('  ', ' ').split()[1])
    oc_value = float(cur_info[10].strip())
    iou_value = float(cur_info[12].strip()[1:-1].replace('  ', ' ').split(' ')[1])
    miou_value = float(cur_info[14].strip())
    fwiou_value = float(cur_info[16].strip())

    return [prec_value, rec_value, f1_value, oc_value, iou_value, miou_value, fwiou_value]


def vis_info(info):
    title_names = 'prec_value, rec_value, f1_value, oc_value, iou_value, miou_value, fwiou_value'.split(', ')
    info = np.asarray(info)
    for i in range(len(title_names)):
        col = list(info[:, i])
        plt.plot([k for k in range(len(col))], col)
        plt.xlabel('epoch')
        plt.ylabel('value')
        plt.title(title_names[i])
        plt.savefig(os.path.join(tgt_path, '{}.png').format(title_names[i]))
        plt.clf()



if __name__ == '__main__':
    config_file = 'config.yaml'

    param_dict = parse_yaml(config_file)

    src_path = os.path.join(param_dict['save_dir_model'], 'val_visual')
    tgt_path = os.path.join(param_dict['save_dir_model'], 'val_curve')
    if os.path.exists(tgt_path) is False:
        os.mkdir(tgt_path)

    dirs = os.listdir(src_path)
    dirs = sorted(dirs, key=lambda x: int(x))
    value_list = []
    for item in tqdm.tqdm(dirs):
        cur_txt = os.path.join(src_path, item, 'accuracy.txt')
        cur_info = parse_txt(cur_txt)
        value_list.append(cur_info)
    vis_info(value_list)
