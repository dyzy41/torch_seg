import cv2
import numpy as np

img = cv2.imread("road.png", 0)

img_ = np.where(img == 255, 0, 255)

cv2.imwrite("road2.png", img_)
