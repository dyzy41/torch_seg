from tools.parse_config_yaml import parse_yaml
import numpy as np
import os
import sys
import  matplotlib.pyplot as plt

# target = sys.argv[1]
# idx = int(sys.argv[2])
target = 'Iou'
idx = 1

tgt2Line = {'OA':0, 'kappa':1, 'mf1-score':2, 'mIou':3, 'precision':5, 'recall':7, 'f1-score':9, 'Iou':11}
yaml_file = '../config.yaml'
param_dict = parse_yaml(yaml_file)
tgt_path = os.path.join(param_dict['save_dir_model'], 'val_visual')
dirs = os.listdir(tgt_path)
dirs = sorted(dirs, key=lambda x:int(x))
save_iter = param_dict['save_iter']
num_list = []
for item in dirs:
    txt_info = open(os.path.join(tgt_path, item, 'accuracy.txt'), 'r').readlines()
    if tgt2Line[target]<4:
        num = float(txt_info[tgt2Line[target]].split('\t')[1])
        num_list.append(num)
    else:
        num = float(txt_info[tgt2Line[target]].strip().split('\t')[idx])
        num_list.append(num)

x = np.asarray([i for i in range(len(num_list))])*int(save_iter)
y = np.asarray(num_list)
plt.plot(x,y,color="red",linewidth=1 )
plt.xlabel("epoch") #xlabel、ylabel：分别设置X、Y轴的标题文字。
plt.ylabel("num")
plt.title("{} curve".format(target)) # title：设置子图的标题。
plt.savefig('target.png',dpi=120,bbox_inches='tight')
print('finshed')