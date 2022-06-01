from __future__ import division
from tools.utils import read_image
import sys
import numpy as np
import os
import yimage
from tools.metrics import get_acc_v2
from timm.optim import create_optimizer_v2
from torch.utils.data import DataLoader
import tqdm
from tools.data_aug import train_aug, val_aug
from tools.dataloader import IsprsSegmentation
import tools
import torch
from tensorboardX import SummaryWriter
from networks.get_model import get_net
from tools.losses import get_loss
from tools.parse_config_yaml import parse_yaml
import torch.onnx


def main():
    train_dataset = IsprsSegmentation(txt_path=param_dict['train_list'], transform=train_aug(param_dict['mean'], param_dict['std']))  # get data
    trainloader = DataLoader(train_dataset, batch_size=param_dict['batch_size'], shuffle=True,
                             num_workers=param_dict['num_workers'], drop_last=True)  # define traindata
    val_dataset = IsprsSegmentation(txt_path=param_dict['val_list'], transform=val_aug(param_dict['mean'], param_dict['std']))  # get data
    valloader = DataLoader(val_dataset, batch_size=param_dict['batch_size'], shuffle=False,
                           num_workers=param_dict['num_workers'])  # define traindata
    start_epoch = 0
    if len(gpu_list) > 1:
        model = torch.nn.DataParallel(frame_work, device_ids=gpu_list)  # use gpu to train
    else:
        model = frame_work
    optimizer = create_optimizer_v2(model, 'adam', lr=param_dict['base_lr'])
    lr_schedule = torch.optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.8)
    if param_dict['resume_ckpt']:
        resume_ckpt = param_dict['resume_ckpt']  # 断点路径
        checkpoint = torch.load(resume_ckpt)  # 加载断点
        model.load_state_dict(checkpoint['net'])  # 加载模型可学习参数
        optimizer.load_state_dict(checkpoint['optimizer'])
        for state in optimizer.state.values():
            for k, v in state.items():
                if torch.is_tensor(v):
                    state[k] = v.to(device)
        start_epoch = checkpoint['epoch']  # 设置开始的epoch
        lr_schedule.load_state_dict(checkpoint['lr_schedule'])  # 加载lr_scheduler
        print('load the model %s' % find_new_file(param_dict['model_dir']))
    model.to(device)

    criterion = get_loss(param_dict['loss_type'])  # define loss
    writer = SummaryWriter(os.path.join(param_dict['save_dir_model'], 'runs'))

    best_val_acc = 0.0
    with open(os.path.join(param_dict['save_dir_model'], 'log.txt'), 'w') as ff:
        for epoch in range(start_epoch, param_dict['epoches']):
            model.train()
            running_loss = 0.0
            batch_num = 0
            for i, data in tqdm.tqdm(enumerate(trainloader)):  # get data
                images, labels = data['image'], data['gt']
                i += images.size()[0]
                labels = labels.long()
                images = images.to(device)
                labels = labels.to(device)
                optimizer.zero_grad()
                outputs = model(images)
                losses = criterion(outputs, labels)  # calculate loss
                losses.backward()  #
                optimizer.step()
                running_loss += losses
                batch_num += images.size()[0]
            print('epoch is {}, train loss is {}'.format(epoch, running_loss.item() / batch_num))
            cur_lr = optimizer.param_groups[0]['lr']
            writer.add_scalar('learning_rate', cur_lr, epoch)
            writer.add_scalar('train_loss', running_loss / batch_num, epoch)
            lr_schedule.step()
            if epoch % param_dict['save_iter'] == 0:
                val_miou, val_acc, val_f1, val_loss = eval(valloader, model, criterion, epoch)
                writer.add_scalar('val_miou', val_miou, epoch)
                writer.add_scalar('val_acc', val_acc, epoch)
                writer.add_scalar('val_f1', val_f1, epoch)
                writer.add_scalar('val_loss', val_loss, epoch)
                cur_log = 'epoch:{}, learning_rate:{}, train_loss:{}, val_loss:{}, val_f1:{}, val_acc:{}, val_miou:{}\n'.format(
                    str(epoch), str(cur_lr), str(running_loss.item() / batch_num), str(val_loss), str(val_f1),
                    str(val_acc),
                    str(val_miou)
                )
                print(cur_log)
                ff.writelines(str(cur_log))
                checkpoint = {
                    "net": model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'lr_schedule': lr_schedule.state_dict(),
                    "epoch": epoch
                }
                if val_miou > best_val_acc:
                    if param_dict['save_mode'] == 'best':
                        torch.save(checkpoint, os.path.join(param_dict['model_dir'], 'valiou_best.pth'))
                    else:
                        torch.save(checkpoint, os.path.join(param_dict['model_dir'], 'valiou_best_{}_{}.pth'.format(epoch, val_miou)))
                    best_val_acc = val_miou
                torch.save(checkpoint, os.path.join(param_dict['model_dir'], 'last_model.pth'))



def eval(valloader, model, criterion, epoch):
    val_num = valloader.dataset.num_sample
    label_all = np.zeros((val_num,) + (param_dict['img_size'], param_dict['img_size']), np.uint8)
    predict_all = np.zeros((val_num,) + (param_dict['img_size'], param_dict['img_size']), np.uint8)
    model.eval()
    if param_dict['val_visual']:
        if os.path.exists(os.path.join(param_dict['save_dir_model'], 'val_visual')) is False:
            os.mkdir(os.path.join(param_dict['save_dir_model'], 'val_visual'))
        if os.path.exists(os.path.join(param_dict['save_dir_model'], 'val_visual', str(epoch))) is False:
            os.mkdir(os.path.join(param_dict['save_dir_model'], 'val_visual', str(epoch)))
            os.mkdir(os.path.join(param_dict['save_dir_model'], 'val_visual', str(epoch), 'slice'))
    with torch.no_grad():
        batch_num = 0
        val_loss = 0.0
        for i, data in tqdm.tqdm(enumerate(valloader), ascii=True, desc="validate step"):  # get data
            images, labels, img_path, gt_path = data['image'], data['gt'], data['img_path'], data['gt_path']
            i += images.size()[0]
            labels = labels.long()
            images = images.to(device)
            labels = labels.to(device)
            if param_dict['extra_loss']:
                outputs, outputs_f, outputs_b = model(images)  # get prediction
            else:
                outputs = model(images)
            vallosses = criterion(outputs, labels)
            if param_dict['loss_type'] == 'triple':
                pred = tools.utils.out2pred(outputs[0], param_dict['num_class'], param_dict['thread'])
            else:
                pred = tools.utils.out2pred(outputs, param_dict['num_class'], param_dict['thread'])
            batch_num += images.size()[0]
            val_loss += vallosses.item()
            if param_dict['val_visual']:
                for kk in range(len(img_path)):
                    cur_name = os.path.basename(img_path[kk])
                    pred_sub = pred[kk, :, :]
                    label_all[i] = read_image(gt_path[kk], 'gt')
                    predict_all[i] = pred_sub
                    yimage.io.write_image(
                        os.path.join(param_dict['save_dir_model'], 'val_visual', str(epoch), 'slice', cur_name),
                        pred_sub,
                        color_table=param_dict['color_table'])
        precision, recall, f1ccore, OA, IoU, mIOU = get_acc_v2(
            label_all, predict_all,
            param_dict['num_class'] + 1 if param_dict['num_class'] == 1 else param_dict['num_class'],
            os.path.join(param_dict['save_dir_model'], 'val_visual', str(epoch)))
        val_loss = val_loss / batch_num
    return IoU[1], OA, f1ccore[1], val_loss


def find_new_file(dir):
    file_lists = os.listdir(dir)
    file_lists.sort(key=lambda fn: os.path.getmtime(dir + fn)
    if not os.path.isdir(dir + fn) else 0)
    if len(file_lists) != 0:
        file = os.path.join(dir, file_lists[-1])
        return file
    else:
        return None


if __name__ == '__main__':
    if len(sys.argv) == 1:
        yaml_file = 'config.yaml'
    else:
        yaml_file = sys.argv[1]
    param_dict = parse_yaml(yaml_file)

    for kv in param_dict.items():
        print(kv)
    os.environ["CUDA_VISIBLE_DEVICES"] = param_dict['gpu_id']
    gpu_list = [i for i in range(len(param_dict['gpu_id'].split(',')))]
    gx = torch.cuda.device_count()
    print('useful gpu count is {}'.format(gx))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    frame_work = get_net(param_dict['model_name'], param_dict['input_bands'], param_dict['num_class'],
                         param_dict['img_size'], param_dict['pretrained_model'])
    if param_dict['vis_graph']:
        sampledata = torch.rand((1, param_dict['input_bands'], param_dict['img_size'], param_dict['img_size']))
        o = frame_work(sampledata)
        onnx_path = os.path.join(param_dict['save_dir_model'], "model_vis.onnx")
        torch.onnx.export(frame_work, sampledata, onnx_path, opset_version=11)
        netron.start(onnx_path)

    if os.path.exists(param_dict['model_dir']) is False:
        os.mkdir(param_dict['model_dir'])
    main()
