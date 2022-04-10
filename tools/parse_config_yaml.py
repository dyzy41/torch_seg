import numpy as np
import yaml
import os
import shutil


def get_base_param(yaml_file):
    f = open(yaml_file, 'r', encoding='utf-8')
    params = yaml.load(f, Loader=yaml.FullLoader)
    return params


def add_param(param_dict):
    cur_path = os.getcwd()

    if param_dict['input_bands'] == 3:
        param_dict['mean'] = (0.472455, 0.320782, 0.318403)
        param_dict['std'] = (0.215084, 0.408135, 0.409993)
    else:
        param_dict['mean'] = (0.472455, 0.320782, 0.318403, 0.357)
        param_dict['std'] = (0.215084, 0.408135, 0.409993, 0.195)
    param_dict['save_dir'] = os.path.join(param_dict['root_path'], '{}_files'.format(param_dict['exp_name']))
    param_dict['save_dir_model'] = os.path.join(param_dict['save_dir'], param_dict['model_name'] + '_' + param_dict['model_experision'])
    if os.path.exists(param_dict['save_dir']) is False:
        os.mkdir(param_dict['save_dir'])
    if os.path.exists(param_dict['save_dir_model']) is False:
        os.mkdir(param_dict['save_dir_model'])
    param_dict['train_list'] = os.path.join(cur_path, '{}/train_list.txt'.format(param_dict['data_name']))
    param_dict['val_list'] = os.path.join(cur_path, '{}/val_list.txt'.format(param_dict['data_name']))
    param_dict['test_list'] = os.path.join(cur_path, '{}/test_list.txt'.format(param_dict['data_name']))
    param_dict['model_dir'] = os.path.join(param_dict['save_dir_model'], './pth_{}/'.format(param_dict['model_name']))
    param_dict['pred_path'] = os.path.join(param_dict['save_dir_model'], param_dict['pred_path'])
    param_dict['pretrained_model'] = os.path.join(param_dict['root_path'], param_dict['pretrained_model'])

    param_dict['train_gt'] = os.path.join(param_dict['root_path'], param_dict['train_gt'])
    param_dict['val_gt'] = os.path.join(param_dict['root_path'], param_dict['val_gt'])
    param_dict['color_table'] = list(np.asarray(param_dict['color_table'].split(',')).astype(np.int).reshape(-1, 3))
    param_dict['color_table'] = [tuple(i) for i in param_dict['color_table']]
    return param_dict


def parse_yaml(yaml_file):
    params = get_base_param(yaml_file)
    params = add_param(params)
    shutil.copy(yaml_file, params['save_dir_model'])
    return params

if __name__ == '__main__':
    f = '../config.yaml'
    params = parse_yaml(f)
    print('ok')
