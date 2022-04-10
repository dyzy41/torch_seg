import os
import yimage
import tqdm


def parse_color_table(color_txt):
    f = open(color_txt, 'r').readlines()[1:]
    color_table = []
    for info in f:
        x = info.split('#')[0].split('/')
        color_table.append((int(x[0]), int(x[1]), int(x[2])))
    return color_table

if __name__ == '__main__':
    p = r'Z:\private\dongsj\0sjcode\code0914_vaiseg\vai_data\gt_nobd'
    tgt = r'Z:\private\dongsj\0sjcode\code0914_vaiseg\vai_data\gt_nobd_2'
    # os.mkdir(tgt)
    labs = os.listdir(p)

    color_txt = r'Z:\private\dongsj\0sjcode\code0916_potseg\pot_data\color_table_isprs.txt'
    color_table = parse_color_table(color_txt)
    for name in tqdm.tqdm(labs):
        cur_lab = yimage.io.read_image(os.path.join(p, name))
        yimage.io.write_image(os.path.join(tgt, name), cur_lab, color_table=color_table)