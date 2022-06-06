import random

import numpy as np
import yaml
import os
import shutil
import json


def get_base_param(yaml_file):
    f = open(yaml_file, 'r', encoding='utf-8')
    params = yaml.load(f, Loader=yaml.FullLoader)
    return params


def gen_color_map(num):
    random.seed(24)
    color_map = []
    while len(color_map) < num:
        r, g, b = np.random.random_integers(0, 255, 3)
        if (r, g, b) not in color_map:
            color_map.append((r, g, b))
    return color_map


def update_param(param_dict):
    cur_path = os.getcwd()

    if param_dict['input_bands'] == 3:
        param_dict['mean'] = (0.413108, 0.447955, 0.441292)
        param_dict['std'] = (0.212406, 0.198994, 0.212655)
    else:
        param_dict['mean'] = (0.472455, 0.320782, 0.318403, 0.357)
        param_dict['std'] = (0.215084, 0.408135, 0.409993, 0.195)
    param_dict['save_dir'] = os.path.join(param_dict['root_path'], '{}_files'.format(param_dict['exp_name']))
    param_dict['save_dir_model'] = os.path.join(param_dict['save_dir'],
                                                param_dict['model_name'] + '_' + param_dict['model_experision'])
    if os.path.exists(param_dict['save_dir']) is False:
        os.mkdir(param_dict['save_dir'])
    if os.path.exists(param_dict['save_dir_model']) is False:
        os.mkdir(param_dict['save_dir_model'])
    param_dict['train_list'] = os.path.join(cur_path, '{}/train_list.txt'.format(param_dict['data_name']))
    param_dict['val_list'] = os.path.join(cur_path, '{}/val_list.txt'.format(param_dict['data_name']))
    param_dict['test_list'] = os.path.join(cur_path, '{}/test_list.txt'.format(param_dict['data_name']))
    param_dict['model_dir'] = os.path.join(param_dict['save_dir_model'], './pth_{}/'.format(param_dict['model_name']))
    param_dict['pred_path'] = os.path.join(param_dict['save_dir_model'], param_dict['pred_path'])
    param_dict['pretrained_model'] = os.path.join(param_dict['pretrained_model'])
    if param_dict['color_table'] == 0:
        param_dict['color_table'] = gen_color_map(param_dict['num_class'])
    else:
        param_dict['color_table'] = list(np.asarray(param_dict['color_table'].split(',')).astype(np.int).reshape(-1, 3))
        param_dict['color_table'] = [tuple(i) for i in param_dict['color_table']]
    return param_dict


def parse_yaml(yaml_file):
    params = get_base_param(yaml_file)
    params = update_param(params)
    shutil.copy(yaml_file, params['save_dir_model'])

    filename = open(os.path.join(params['save_dir_model'], 'param.txt'), 'w')  # dict转txt
    for k, v in params.items():
        filename.write(k + ':' + str(v))
        filename.write('\n')
    filename.close()
    return params


if __name__ == '__main__':
    f = '../config.yaml'
    params = parse_yaml(f)
    print('ok')
