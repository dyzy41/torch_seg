import os
import shutil
import cv2
import tqdm

p = r'U:\private\dongsj\CUG_seg\CHN6-CUG\val\images'
p2 = r'U:\private\dongsj\CUG_seg\CHN6-CUG\val\black'
imgs = os.listdir(p)

for item in tqdm.tqdm(imgs):
    try:
        img = cv2.imread(os.path.join(p, item), 0)
        if max(img.flatten()) == 0:
            shutil.move(os.path.join(p, item), os.path.join(p2, item))
    except:
        pass
