import os
import random

pimg = r'/nfs/project/netdisk/192.168.10.224/d/private/dongsj/CUG_seg/CHN6-CUG/train/images'
plab = r'/nfs/project/netdisk/192.168.10.224/d/private/dongsj/CUG_seg/CHN6-CUG/train/gt'
p_list = './data/train_list.txt'
imgs = os.listdir(pimg)
random.shuffle(imgs)

with open(p_list, 'w') as f:
    for item in imgs:
        img = os.path.join(pimg, item)
        lab = os.path.join(plab, item.replace('_sat.jpg', '_mask.png'))
        info = '{}  {}\n'.format(img, lab)
        f.writelines(info)

# pimg = r'/nfs/project/netdisk/192.168.10.224/d/private/dongsj/CUG_seg/CHN6-CUG/val/images'
# plab = r'/nfs/project/netdisk/192.168.10.224/d/private/dongsj/CUG_seg/CHN6-CUG/val/gt'
# p_list = './data/val_list.txt'
# imgs = os.listdir(pimg)
#
# with open(p_list, 'w') as f:
#     for item in imgs:
#         img = os.path.join(pimg, item)
#         lab = os.path.join(plab, item.replace('_sat.jpg', '_mask.png'))
#         info = '{}  {}\n'.format(img, lab)
#         f.writelines(info)
