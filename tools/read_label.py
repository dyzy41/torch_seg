import cv2
import numpy as np
from PIL import Image
import yimage
import os

p = r'E:\dataset\smallcity\masks'
imgs = os.listdir(p)
for i in range(len(imgs)):
    img = yimage.io.read_image(os.path.join(p, imgs[i]))
    print(set(img.flatten()))


