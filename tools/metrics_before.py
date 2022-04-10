import os
import yimage
import numpy as np
from sklearn.metrics import f1_score, confusion_matrix
import tqdm
# from collections import Counter

def cal_acc(pred, gt):
    pixel_num = pred.shape[0] * pred.shape[1]
    # pred[gt == 0] = 0
    true_num = np.sum(gt == pred)
    pixel_acc = true_num / pixel_num
    return pixel_acc, true_num, pixel_num


def cal_f1(pred, gt):
    # pred[gt == 0] = 0
    f1 = f1_score(gt.flatten(), pred.flatten(), average=None)
    # cm = confusion_matrix(gt.flatten(), pred.flatten(), labels=[i for i in range(num_class)])
    cm = confusion_matrix(gt.flatten(), pred.flatten())
    return f1, cm


def metrics(confu_mat_total, save_path):
    '''
    :param confu_mat: 总的混淆矩阵
    backgound：是否干掉背景
    :return: txt写出混淆矩阵, precision，recall，IOU，f-score
    '''
    class_num = confu_mat_total.shape[0]
    confu_mat = confu_mat_total.astype(np.float32) + 0.0001
    col_sum = np.sum(confu_mat, axis=1)  # 按行求和
    raw_sum = np.sum(confu_mat, axis=0)  # 每一列的数量

    '''计算各类面积比，以求OA值'''
    oa = 0
    for i in range(class_num):
        oa = oa + confu_mat[i, i]
    oa = oa / confu_mat.sum()

    '''Kappa'''
    pe_fz = 0
    for i in range(class_num):
        pe_fz += col_sum[i] * raw_sum[i]
    pe = pe_fz / (np.sum(confu_mat) * np.sum(confu_mat))
    kappa = (oa - pe) / (1 - pe)

    # 将混淆矩阵写入excel中
    TP = []  # 识别中每类分类正确的个数

    for i in range(class_num):
        TP.append(confu_mat[i, i])

    # 计算f1-score
    TP = np.array(TP)
    FN = col_sum - TP
    FP = raw_sum - TP

    # 计算并写出precision，recall, f1-score，f1-m以及mIOU

    f1_m = []
    iou_m = []
    for i in range(class_num):
        # 写出f1-score
        f1 = TP[i] * 2 / (TP[i] * 2 + FP[i] + FN[i])
        f1_m.append(f1)
        iou = TP[i] / (TP[i] + FP[i] + FN[i])
        iou_m.append(iou)

    f1_m = np.array(f1_m)
    iou_m = np.array(iou_m)
    if save_path is not None:
        with open(save_path + 'accuracy.txt', 'w') as f:
            f.write('OA:\t%.4f\n' % (oa * 100))
            f.write('kappa:\t%.4f\n' % (kappa * 100))
            f.write('mf1-score:\t%.4f\n' % (np.mean(f1_m) * 100))
            f.write('mIou:\t%.4f\n' % (np.mean(iou_m) * 100))

            # 写出precision
            f.write('precision:\n')
            for i in range(class_num):
                f.write('%.4f\t' % (float(TP[i] / raw_sum[i]) * 100))
            f.write('\n')

            # 写出recall
            f.write('recall:\n')
            for i in range(class_num):
                f.write('%.4f\t' % (float(TP[i] / col_sum[i]) * 100))
            f.write('\n')

            # 写出f1-score
            f.write('f1-score:\n')
            for i in range(class_num):
                f.write('%.4f\t' % (float(f1_m[i]) * 100))
            f.write('\n')

            # 写出 IOU
            f.write('Iou:\n')
            for i in range(class_num):
                f.write('%.4f\t' % (float(iou_m[i]) * 100))
            f.write('\n')
    return f1_m, iou_m

def get_acc_info(p_pred, p_gt, num_class=2, save_path='./'):
    # gts = sorted(os.listdir(p_gt))
    preds = sorted(os.listdir(p_pred))
    up_all = []
    down_all = []
    cm_init = np.zeros((num_class, num_class))
    for name in tqdm.tqdm(preds):
        print("******************{}******************".format(name))
        pred = yimage.io.read_image(os.path.join(p_pred, name))
        gt = yimage.io.read_image(os.path.join(p_gt, name))
        gt = np.where(gt > 0, 1, gt)
        # pred = np.where(pred > 0, 1, pred)
        acc, up, down = cal_acc(pred, gt)
        f1, cm = cal_f1(pred, gt)
        up_all.append(up)
        down_all.append(down)
        cm_init += cm
        print('file is {}, acc is {}\n, f1 is'.format(name, acc))
        print(f1)
    f1_m, iou_m = metrics(cm_init, save_path)
    acc_all = np.sum(up_all) / np.sum(down_all)
    print('####acc is {}'.format(acc_all))
    print(f1_m)
    print(iou_m)
    return f1_m, iou_m, acc_all


if __name__ == '__main__':
    p_gt = r'U:\private\dongsj\CUG_seg\CHN6-CUG\train\gt'
    p_pred = r'U:\private\dongsj\CUG_seg\acc_test\pred_'
    num_class = 2
    res = get_acc_info(p_pred, p_gt, num_class)
    print(res)

