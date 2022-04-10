import yimage
from PIL import Image
import os
import cv2
import tqdm
from tools.utils import label_mapping, parse_color_table
import sys


p = r'E:\0Ecode\code220329_stoneseg\dataset\mask\masks'
save = r'E:\0Ecode\code220329_stoneseg\dataset\mask\masks_vis'

names = os.listdir(p)

for item in tqdm.tqdm(names):
    img = yimage.io.read_image(os.path.join(p, item))

    color_table = [(0,0,0), (255,255,255)]
    yimage.io.write_image(os.path.join(save, item),
        img,
        color_table=color_table)
