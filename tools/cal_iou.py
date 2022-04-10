import os
import tqdm
import numpy as np
from sklearn.metrics import f1_score, accuracy_score, classification_report
import yimage


# def accuracy(preds, label):
#     valid = (label >= 0)
#     acc_sum = (valid * (preds == label)).sum()
#     valid_sum = valid.sum()
#     acc = float(acc_sum) / (valid_sum + 1e-10)
#     return acc, valid_sum

def accuracy(preds, label):
    valid = (label >= 0)
    acc_sum = (valid * (preds == label)).sum()
    valid_sum = valid.sum()
    acc = float(acc_sum) / (valid_sum - np.where(label == 255, 1, 0).sum() + 1e-13)
    return acc, valid_sum


def intersectionAndUnion(imPred, imLab, numClass):
    imPred = np.asarray(imPred).copy()
    imLab = np.asarray(imLab).copy()

    imPred += 1
    imLab += 1
    # Remove classes from unlabeled pixels in gt image.
    # We should not penalize detections in unlabeled portions of the image.
    imPred = imPred * (imLab > 0)

    # Compute area intersection:
    intersection = imPred * (imPred == imLab)
    (area_intersection, _) = np.histogram(
        intersection, bins=numClass, range=(1, numClass))

    # Compute area union:
    (area_pred, _) = np.histogram(imPred, bins=numClass, range=(1, numClass))
    (area_lab, _) = np.histogram(imLab, bins=numClass, range=(1, numClass))
    area_union = area_pred + area_lab - area_intersection

    return (area_intersection, area_union)


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.initialized = False
        self.val = None
        self.avg = None
        self.sum = None
        self.count = None

    def initialize(self, val, weight):
        self.val = val
        self.avg = val
        self.sum = val * weight
        self.count = weight
        self.initialized = True

    def update(self, val, weight=1):
        if not self.initialized:
            self.initialize(val, weight)
        else:
            self.add(val, weight)

    def add(self, val, weight):
        self.val = val
        self.sum += val * weight
        self.count += weight
        self.avg = self.sum / self.count

    def value(self):
        return self.val

    def average(self):
        return self.avg


def evaluate(val_gt, pred_dir, num_class):
    acc_meter = AverageMeter()
    intersection_meter = AverageMeter()
    union_meter = AverageMeter()
    names = os.listdir(pred_dir)
    # pred_all = []
    # gt_all = []
    for name in tqdm.tqdm(names):
        pred = yimage.io.read_image(os.path.join(pred_dir, name))
        gt = yimage.io.read_image(os.path.join(val_gt, name))
        gt = np.where(gt > 0, 1, 0)
        acc, pix = accuracy(pred, gt)
        # pred_all += list(pred.flatten())
        # gt_all += list(gt.flatten())
        intersection, union = intersectionAndUnion(pred, gt, num_class)
        acc_meter.update(acc, pix)
        intersection_meter.update(intersection)
        union_meter.update(union)
    # f1 = f1_score(gt_all, pred_all, average='macro')
    iou = intersection_meter.sum / (union_meter.sum + 1e-10)
    for i, _iou in enumerate(iou):
        print('class [{}], IoU: {}'.format(i, _iou))
    # print(classification_report(gt_all, pred_all))
    print('[Eval Summary]:')
    print('Mean IoU: {:.4}, Accuracy: {:.4f}%'
          .format(iou.mean(), acc_meter.average() * 100))
    return iou.mean(), acc_meter.average()


if __name__ == '__main__':
    p = r'U:\private\dongsj\CUG_seg\result2'
    gt = r'U:\private\dongsj\CUG_seg\CHN6-CUG\val\gt'
    evaluate(gt, p, 2)
