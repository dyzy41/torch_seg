import os
import pydicom
import numpy
from matplotlib import pyplot

# 用lstFilesDCM作为存放DICOM files的列表
PathDicom = "./dataset/label_data/ILL"  # 与python文件同一个目录下的文件夹
lstFilesDCM = []

for dirName, subdirList, fileList in os.walk(PathDicom):
    for filename in fileList:
        if ".dcm" in filename.lower():  # 判断文件是否为dicom文件
            print(filename)
            lstFilesDCM.append(os.path.join(dirName, filename))  # 加入到列表中

## 将第一张图片作为参考图
RefDs = pydicom.read_file(lstFilesDCM[0])  # 读取第一张dicom图片

# 建立三维数组
ConstPixelDims = (int(RefDs.Rows), int(RefDs.Columns), len(lstFilesDCM))

# 得到spacing值 (mm为单位)
ConstPixelSpacing = (float(RefDs.PixelSpacing[0]), float(RefDs.PixelSpacing[1]), float(RefDs.SliceThickness))

# 三维数据
x = numpy.arange(0.0, (ConstPixelDims[0] + 1) * ConstPixelSpacing[0],
                 ConstPixelSpacing[0])  # 0到（第一个维数加一*像素间的间隔），步长为constpixelSpacing
y = numpy.arange(0.0, (ConstPixelDims[1] + 1) * ConstPixelSpacing[1], ConstPixelSpacing[1])  #
z = numpy.arange(0.0, (ConstPixelDims[2] + 1) * ConstPixelSpacing[2], ConstPixelSpacing[2])  #

ArrayDicom = numpy.zeros(ConstPixelDims, dtype=RefDs.pixel_array.dtype)

# 遍历所有的dicom文件，读取图像数据，存放在numpy数组中
for filenameDCM in lstFilesDCM:
    ds = pydicom.read_file(filenameDCM)
    ArrayDicom[:, :, lstFilesDCM.index(filenameDCM)] = ds.pixel_array

# 轴状面显示
pyplot.figure(dpi=300)
pyplot.axes().set_aspect('equal', 'datalim')
pyplot.set_cmap(pyplot.gray())
pyplot.pcolormesh(x, y, numpy.flipud(ArrayDicom[:, :, 9]))  # 第三个维度表示现在展示的是第几层
pyplot.show()
"""
# 冠状面显示
pyplot.figure(dpi=300)
pyplot.axes().set_aspect('equal', 'datalim')
pyplot.set_cmap(pyplot.gray())
pyplot.pcolormesh(z, x, numpy.flipud(ArrayDicom[:, 150, :]))
pyplot.show()
"""