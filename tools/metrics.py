import numpy as np
import cv2
import os
from PIL import Image
import yimage
from tools.utils import read_image



#  获取颜色字典
#  labelFolder 标签文件夹,之所以遍历文件夹是因为一张标签可能不包含所有类别颜色
#  classNum 类别总数(含背景)
def color_dict(labelFolder, classNum):
    colorDict = []
    #  获取文件夹内的文件名
    ImageNameList = os.listdir(labelFolder)
    for i in range(len(ImageNameList)):
        ImagePath = labelFolder + "/" + ImageNameList[i]
        img = cv2.imread(ImagePath).astype(np.uint32)
        #  如果是灰度，转成RGB
        if len(img.shape) == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB).astype(np.uint32)
        #  为了提取唯一值，将RGB转成一个数
        img_new = img[:, :, 0] * 1000000 + img[:, :, 1] * 1000 + img[:, :, 2]
        unique = np.unique(img_new)
        #  将第i个像素矩阵的唯一值添加到colorDict中
        for j in range(unique.shape[0]):
            colorDict.append(unique[j])
        #  对目前i个像素矩阵里的唯一值再取唯一值
        colorDict = sorted(set(colorDict))
        #  若唯一值数目等于总类数(包括背景)ClassNum，停止遍历剩余的图像
        if (len(colorDict) == classNum):
            break
    #  存储颜色的BGR字典，用于预测时的渲染结果
    colorDict_BGR = []
    for k in range(len(colorDict)):
        #  对没有达到九位数字的结果进行左边补零(eg:5,201,111->005,201,111)
        color = str(colorDict[k]).rjust(9, '0')
        #  前3位B,中3位G,后3位R
        color_BGR = [int(color[0: 3]), int(color[3: 6]), int(color[6: 9])]
        colorDict_BGR.append(color_BGR)
    #  转为numpy格式
    colorDict_BGR = np.array(colorDict_BGR)
    #  存储颜色的GRAY字典，用于预处理时的onehot编码
    colorDict_GRAY = colorDict_BGR.reshape((colorDict_BGR.shape[0], 1, colorDict_BGR.shape[1])).astype(np.uint8)
    colorDict_GRAY = cv2.cvtColor(colorDict_GRAY, cv2.COLOR_BGR2GRAY)
    return colorDict_BGR, colorDict_GRAY


def ConfusionMatrix(numClass, imgPredict, Label):
    #  返回混淆矩阵
    mask = (Label >= 0) & (Label < numClass)
    label = numClass * Label[mask] + imgPredict[mask]
    count = np.bincount(label, minlength=numClass ** 2)
    confusionMatrix = count.reshape(numClass, numClass)
    return confusionMatrix


def OverallAccuracy(confusionMatrix):
    #  返回所有类的整体像素精度OA
    # acc = (TP + TN) / (TP + TN + FP + TN)
    OA = np.diag(confusionMatrix).sum() / confusionMatrix.sum()
    return OA


def Precision(confusionMatrix):
    #  返回所有类别的精确率precision
    precision = np.diag(confusionMatrix) / confusionMatrix.sum(axis=0)
    return precision


def Recall(confusionMatrix):
    #  返回所有类别的召回率recall
    recall = np.diag(confusionMatrix) / confusionMatrix.sum(axis=1)
    return recall


def F1Score(confusionMatrix):
    precision = np.diag(confusionMatrix) / confusionMatrix.sum(axis=0)
    recall = np.diag(confusionMatrix) / confusionMatrix.sum(axis=1)
    f1score = 2 * precision * recall / (precision + recall)
    return f1score


def IntersectionOverUnion(confusionMatrix):
    #  返回交并比IoU
    intersection = np.diag(confusionMatrix)
    union = np.sum(confusionMatrix, axis=1) + np.sum(confusionMatrix, axis=0) - np.diag(confusionMatrix)
    IoU = intersection / union
    return IoU


def MeanIntersectionOverUnion(confusionMatrix):
    #  返回平均交并比mIoU
    intersection = np.diag(confusionMatrix)
    union = np.sum(confusionMatrix, axis=1) + np.sum(confusionMatrix, axis=0) - np.diag(confusionMatrix)
    IoU = intersection / union
    mIoU = np.nanmean(IoU)
    return mIoU


def Frequency_Weighted_Intersection_over_Union(confusionMatrix):
    #  返回频权交并比FWIoU
    freq = np.sum(confusionMatrix, axis=1) / np.sum(confusionMatrix)
    iu = np.diag(confusionMatrix) / (
            np.sum(confusionMatrix, axis=1) +
            np.sum(confusionMatrix, axis=0) -
            np.diag(confusionMatrix))
    FWIoU = (freq[freq > 0] * iu[freq > 0]).sum()
    return FWIoU


def get_acc_v2(label_all, predict_all, classNum=2, save_path='./'):
    label_all = label_all.flatten()
    predict_all = predict_all.flatten()
    confusionMatrix = ConfusionMatrix(classNum, predict_all, label_all)
    precision = Precision(confusionMatrix)
    recall = Recall(confusionMatrix)
    OA = OverallAccuracy(confusionMatrix)
    IoU = IntersectionOverUnion(confusionMatrix)
    FWIOU = Frequency_Weighted_Intersection_over_Union(confusionMatrix)
    mIOU = MeanIntersectionOverUnion(confusionMatrix)
    f1ccore = F1Score(confusionMatrix)
    print("")
    print("confusion_matrix:")
    print(confusionMatrix)
    print("precision:")
    print(precision)
    print("recall:")
    print(recall)
    print("F1-Score:")
    print(f1ccore)
    print("overall_accuracy:")
    print(OA)
    print("IoU:")
    print(IoU)
    print("mIoU:")
    print(mIOU)
    print("FWIoU:")
    print(FWIOU)
    with open('{}/accuracy.txt'.format(save_path), 'w') as ff:
        ff.writelines("confusion_matrix:\n")
        ff.writelines(str(confusionMatrix)+"\n")
        ff.writelines("precision:\n")
        ff.writelines(str(precision)+"\n")
        ff.writelines("recall:\n")
        ff.writelines(str(recall)+"\n")
        ff.writelines("F1-Score:\n")
        ff.writelines(str(f1ccore)+"\n")
        ff.writelines("overall_accuracy:\n")
        ff.writelines(str(OA)+"\n")
        ff.writelines("IoU:\n")
        ff.writelines(str(IoU)+"\n")
        ff.writelines("mIoU:\n")
        ff.writelines(str(mIOU)+"\n")
        ff.writelines("FWIoU:\n")
        ff.writelines(str(FWIOU)+"\n")
    return precision, recall, f1ccore, OA, IoU, mIOU



def get_acc_info(PredictPath, LabelPath, classNum=2, save_path='./'):
    #  获取类别颜色字典
    colorDict_BGR, colorDict_GRAY = color_dict(LabelPath, classNum)

    #  获取文件夹内所有图像
    labelList = os.listdir(LabelPath)
    PredictList = os.listdir(PredictPath)

    #  读取第一个图像，后面要用到它的shape
    Label0 = cv2.imread(PredictPath + "//" + PredictList[0], 0)

    #  图像数目
    label_num = len(PredictList)

    #  把所有图像放在一个数组里
    label_all = np.zeros((label_num,) + Label0.shape, np.uint8)
    predict_all = np.zeros((label_num,) + Label0.shape, np.uint8)
    for i in range(label_num):
        Label = read_image(LabelPath + "//" + PredictList[i], 'gt')
        label_all[i] = Label
        Predict = read_image(PredictPath + "//" + PredictList[i])
        predict_all[i] = Predict

    #  把颜色映射为0,1,2,3...
    for i in range(colorDict_GRAY.shape[0]):
        label_all[label_all == colorDict_GRAY[i][0]] = i
        predict_all[predict_all == colorDict_GRAY[i][0]] = i

    #  拉直成一维
    label_all = label_all.flatten()
    predict_all = predict_all.flatten()

    #  计算混淆矩阵及各精度参数
    confusionMatrix = ConfusionMatrix(classNum, predict_all, label_all)
    precision = Precision(confusionMatrix)
    recall = Recall(confusionMatrix)
    OA = OverallAccuracy(confusionMatrix)
    IoU = IntersectionOverUnion(confusionMatrix)
    FWIOU = Frequency_Weighted_Intersection_over_Union(confusionMatrix)
    mIOU = MeanIntersectionOverUnion(confusionMatrix)
    f1ccore = F1Score(confusionMatrix)

    for i in range(colorDict_BGR.shape[0]):
        #  输出类别颜色,需要安装webcolors,直接pip install webcolors
        try:
            import webcolors

            rgb = colorDict_BGR[i]
            rgb[0], rgb[2] = rgb[2], rgb[0]
            print(webcolors.rgb_to_name(rgb), end="  ")
        #  不安装的话,输出灰度值
        except:
            print(colorDict_GRAY[i][0], end="  ")
    print("")
    print("confusion_matrix:")
    print(confusionMatrix)
    print("precision:")
    print(precision)
    print("recall:")
    print(recall)
    print("F1-Score:")
    print(f1ccore)
    print("overall_accuracy:")
    print(OA)
    print("IoU:")
    print(IoU)
    print("mIoU:")
    print(mIOU)
    print("FWIoU:")
    print(FWIOU)
    with open('{}/accuracy.txt'.format(save_path), 'w') as ff:
        ff.writelines("confusion_matrix:\n")
        ff.writelines(str(confusionMatrix)+"\n")
        ff.writelines("precision:\n")
        ff.writelines(str(precision)+"\n")
        ff.writelines("recall:\n")
        ff.writelines(str(recall)+"\n")
        ff.writelines("F1-Score:\n")
        ff.writelines(str(f1ccore)+"\n")
        ff.writelines("overall_accuracy:\n")
        ff.writelines(str(OA)+"\n")
        ff.writelines("IoU:\n")
        ff.writelines(str(IoU)+"\n")
        ff.writelines("mIoU:\n")
        ff.writelines(str(mIOU)+"\n")
        ff.writelines("FWIoU:\n")
        ff.writelines(str(FWIOU)+"\n")
    return precision, recall, f1ccore, OA, IoU, mIOU


if __name__ == '__main__':
    #################################################################
    #  标签图像文件夹
    LabelPath = r'U:\private\dongsj\CUG_seg\CHN6-CUG\train\gt'
    #  预测图像文件夹
    PredictPath = r'U:\private\dongsj\CUG_seg\1109_files\FANet50_v5_v1\val_visual\268\slice'
    #  类别数目(包括背景)
    classNum = 2
    #################################################################
    precision, recall, f1ccore, OA, IoU, mIOU = get_acc_info(PredictPath, LabelPath, classNum)
