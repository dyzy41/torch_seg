import argparse
import json,glob
import os,cv2
import os.path as osp
import warnings
import copy
import numpy as np
import PIL.Image
from skimage import io
import yaml
from labelme import utils

"""
文件存储格式
--outdir
        |--imgs       原图
        |--masks      掩码图
        |--yml
        |--txt
"""
#todo  生成通道为1的掩码图,并以png保存
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-j", '--json_file', default=r"E:\0Ecode\code220329_stoneseg\dataset\json_files",type=str, help='json file')  # 标注文件json所在的文件夹
    parser.add_argument('-o', '--outdir',default=r"E:\0Ecode\code220329_stoneseg\dataset\mask", type=str, help='save path')
    args = parser.parse_args()

    json_file = args.json_file
    outdir=args.outdir

    list = glob.glob(os.path.join(json_file,"*.json"))[0:]   # 获取json文件列表
    for i in range(0, len(list)):
        path = list[i]  # 获取每个json文件的绝对路径
        filename = list[i][:-5]       # 提取出.json前的字符作为文件名，以便后续保存Label图片的时候使用
        extension = list[i][-4:]
        try:
            if extension == 'json':
                if os.path.isfile(path):
                    data = json.load(open(path))
                    # 根据'imageData'字段的字符可以得到原图像,json是以base编码存储的图片
                    # img = utils.image.img_b64_to_arr(data['imageData'])
                    img = cv2.cvtColor(cv2.imread(filename + ".jpg"), cv2.COLOR_BGR2RGB)

                    # lbl为label图片（标注的地方用类别名对应的数字来标，其他为0）lbl_names为label名和数字的对应关系字典
                    lbl, lbl_names = utils.labelme_shapes_to_label(img.shape, data[
                        'shapes'])  # data['shapes']是json文件中记录着标注的位置及label等信息的字段

                    # captions = ['%d: %s' % (l, name) for l, name in enumerate(lbl_names)]
                    # lbl_viz = utils.draw.draw_label(lbl, img, captions)
                    out_dir_name = os.path.basename(list[i])[:-5]
                    out_dir = osp.join(outdir)

                    for path in [out_dir, osp.join(out_dir, "imgs"), osp.join(out_dir, "masks")]:
                        if not osp.exists(path):
                            os.mkdir(path)

                    PIL.Image.fromarray(img).save(osp.join(out_dir, '{}/{}.png'.format("imgs", out_dir_name)))
                    PIL.Image.fromarray(lbl).save(osp.join(out_dir, '{}/{}.png'.format("masks", out_dir_name)))
                    # PIL.Image.fromarray(lbl_viz).save(osp.join(out_dir, '{}_viz.jpg'.format(filename)))

                    with open(osp.join(out_dir, 'label_names.txt'), 'w') as f:
                        for lbl_name in lbl_names:
                            f.write(lbl_name + '\n')

                    warnings.warn('info.yaml is being replaced by label_names.txt')
                    info = dict(label_names=lbl_names)
                    with open(osp.join(out_dir, 'info.yaml'), 'w') as f:
                        yaml.safe_dump(info, f, default_flow_style=False)

                    print('Saved to: %s' % out_dir)
        except:
            pass


if __name__ == '__main__':
    main()