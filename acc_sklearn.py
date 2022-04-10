import os
import numpy as np
import yimage
from sklearn.metrics import classification_report

p1 = r'U:\private\dongsj\CUG_seg\acc_test\pred_'
p2 = r'U:\private\dongsj\CUG_seg\CHN6-CUG\train\gt'
labs = os.listdir(p1)
pred_all = []
gt_all = []
for item in labs:
    pred = yimage.io.read_image(os.path.join(p1, item))
    gt = yimage.io.read_image(os.path.join(p2, item))
    gt = np.where(gt>0, 1, 0)
    pred_all += list(pred.flatten())
    gt_all += list(gt.flatten())

print(classification_report(gt_all, pred_all, digits=6))

#               precision    recall  f1-score   support
#
#            0   0.980065  0.970690  0.975355   4751757
#            1   0.740442  0.808966  0.773189    491123
#
#     accuracy                       0.955541   5242880
#    macro avg   0.860253  0.889828  0.874272   5242880
# weighted avg   0.957618  0.955541  0.956417   5242880