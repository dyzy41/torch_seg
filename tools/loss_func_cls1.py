import functools
import torch.nn as nn
import numpy as np
import torch
import torch.nn.functional as F


class SSIM_loss(nn.Module):
    def __init__(self, alpha=0.1, betta=0.2, gamma=0.3):
        super(SSIM_loss, self).__init__()
        self.alpha = alpha
        self.betta = betta
        self.gamma = gamma

    def forward(self, input, target):
        input = input.to('cpu')
        target = target.to('cpu')

        target = torch.sigmoid(target)

        mu_pred = target.sum(2).sum(2).reshape(-1)
        mu_real = input.sum(2).sum(2).reshape(-1)

        sig_pred = torch.var(target, dim=(2, 3)).reshape(-1)
        sig_real = torch.var(input, dim=(2, 3)).reshape(-1)
        sig_real_pred = torch.var(input * target, dim=(2, 3)).reshape(-1)
        sig_pred = torch.sqrt(sig_pred)
        sig_real = torch.sqrt(sig_real)
        sig_real_pred = torch.sqrt(sig_real_pred)

        l = (2 * mu_pred * mu_real + 1) / (mu_pred ** 2 + mu_real ** 2 + 1)
        c = (2 * sig_pred * sig_real + 1) / (sig_pred ** 2 + sig_real ** 2 + 1)
        s = (sig_real_pred + 1) / (sig_pred * sig_real + 1)

        loss = (l ** self.alpha) * (c ** self.betta) * (s ** self.gamma)
        return torch.mean(1 - loss)


class FocalLoss(nn.Module):
    def __init__(self, eps=1e-8, gamma=2):
        super(FocalLoss, self).__init__()
        self.eps = eps
        self.gamma = gamma

    def forward(self, input, target):
        input = torch.clamp(input, min=self.eps, max=1 - self.eps)
        input = torch.sigmoid(input)
        target = torch.clamp(target, min=self.eps)
        your_loss = -torch.sum(
            ((1 - input) ** self.gamma) * target * torch.log(input) + (input ** self.gamma) * (1 - target) * torch.log(
                1 - input))

        return your_loss


class DiceLoss(nn.Module):
    def __init__(self,):
        super(DiceLoss, self).__init__()

    def forward(self, input, target):
        input = torch.sigmoid(input)
        num = (2 * target * input).sum()
        den = (target + input).sum()
        res = 1 - (num + 1) / (den + 1)
        return res

# def SSIM_loss(y_real, y_pred, alpha=0.1, betta=0.2, gamma=0.3):
#     y_real = y_real.to('cpu')
#     y_pred = y_pred.to('cpu')
#
#     y_pred = torch.sigmoid(y_pred)
#
#     mu_pred = y_pred.sum(2).sum(2).reshape(-1)
#     mu_real = y_real.sum(2).sum(2).reshape(-1)
#
#     sig_pred = torch.var(y_pred, dim=(2,3)).reshape(-1)
#     sig_real = torch.var(y_real, dim=(2,3)).reshape(-1)
#     sig_real_pred = torch.var(y_real*y_pred, dim=(2,3)).reshape(-1)
#     sig_pred = torch.sqrt(sig_pred)
#     sig_real = torch.sqrt(sig_real)
#     sig_real_pred = torch.sqrt(sig_real_pred)
#
#     l = (2*mu_pred*mu_real+1)/(mu_pred**2+mu_real**2+1)
#     c = (2*sig_pred*sig_real+1)/(sig_pred**2+sig_real**2+1)
#     s = (sig_real_pred+1)/(sig_pred*sig_real+1)
#
#     loss = (l**alpha)*(c**betta)*(s**gamma)
#     return torch.mean(1-loss)


# def FocalLoss(y_real, y_pred, eps=1e-8, gamma=2):
#     y_pred = torch.clamp(y_pred, min=eps, max=1 - eps)
#     y_pred = torch.sigmoid(y_pred)
#     y_real = torch.clamp(y_real, min=eps)
#     your_loss = -torch.sum(
#         ((1 - y_pred) ** gamma) * y_real * torch.log(y_pred) + (y_pred ** gamma) * (1 - y_real) * torch.log(1 - y_pred))
#
#     return your_loss


# def DiceLoss(y_real, y_pred):
#     y_pred = torch.sigmoid(y_pred)
#     num = (2 * y_real * y_pred).sum()
#     den = (y_real + y_pred).sum()
#     res = 1 - (num + 1) / (den + 1)
#     return res


class TriLoss(nn.Module):
    def __init__(self, weight=None, reduction='mean'):
        super(TriLoss, self).__init__()
        weights_f = torch.tensor([0.01, 1]).cuda().float()
        weights_b = torch.tensor([1, 0.01]).cuda().float()
        self.cef = nn.CrossEntropyLoss(weight=weights_f, reduction=reduction)
        self.ceb = nn.CrossEntropyLoss(weight=weights_b, reduction=reduction)
        self.BCE = nn.BCELoss()

    def forward(self, input, target):
        out, out_b, out_f = input
        out = out.squeeze(1)
        losses1 = self.BCE(out, target.float())  # calculate loss
        losses_f = self.cef(out_f, target)  # calculate loss
        losses_b = self.ceb(out_b, target)  # calculate loss
        losses = losses1 + losses_b * 0.3 + losses_f * 0.7
        return losses


class BinaryDiceLoss(nn.Module):
    """Dice loss of binary class
    Args:
        smooth: A float number to smooth loss, and avoid NaN error, default: 1
        p: Denominator value: \sum{x^p} + \sum{y^p}, default: 2
        predict: A tensor of shape [N, *]
        target: A tensor of shape same with predict
        reduction: Reduction method to apply, return mean over batch if 'mean',
            return sum if 'sum', return a tensor of shape [N,] if 'none'
    Returns:
        Loss tensor according to arg reduction
    Raise:
        Exception if unexpected reduction
    """

    def __init__(self, weight=None, smooth=1, p=2, reduction='mean'):
        super(BinaryDiceLoss, self).__init__()
        self.smooth = smooth
        self.p = p
        self.reduction = reduction

    def forward(self, input, target):
        assert input.shape[0] == target.shape[0], "input & target batch size don't match"
        input = input.contiguous().view(input.shape[0], -1)
        target = target.contiguous().view(target.shape[0], -1)

        num = torch.sum(torch.mul(input, target), dim=1) + self.smooth
        den = torch.sum(input.pow(self.p) + target.pow(self.p), dim=1) + self.smooth

        loss = 1 - num / den

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        elif self.reduction == 'none':
            return loss
        else:
            raise Exception('Unexpected reduction {}'.format(self.reduction))


class BCELoss(nn.Module):
    def __init__(self, smooth=1., ignore_index=255):
        super(BCELoss, self).__init__()
        self.ignore_index = ignore_index
        self.smooth = smooth
        self.BCE = nn.BCELoss()

    def forward(self, input, target):
        input = F.sigmoid(input)
        input = input.squeeze(1)
        loss = self.BCE(input, target.float())

        return loss


class BCE_BDLoss(nn.Module):
    def __init__(self, weight=None, loss_weight=None):
        super(BCE_BDLoss, self).__init__()
        self.BCE = nn.BCELoss()
        self.loss_weight = loss_weight
        self.edge_conv = EdgeConv().cuda()

    def forward(self, input, target):
        input = input.squeeze(1)
        loss = self.BCE(input, target.float())
        input_bd = self.edge_conv(input)
        target_bd = self.edge_conv(target.float())
        loss_bd = BoundaryLoss(input_bd, target_bd)
        return loss * self.loss_weight[0] + loss_bd * self.loss_weight[1]


class EdgeConv(nn.Module):
    def __init__(self):
        super(EdgeConv, self).__init__()
        self.conv_op = nn.Conv2d(1, 1, kernel_size=3, padding=1, bias=False)
        sobel_kernel = np.array([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]], dtype='float32')
        sobel_kernel = sobel_kernel.reshape((1, 1, 3, 3))
        self.conv_op.weight.data = torch.from_numpy(sobel_kernel)

    def forward(self, x):
        x = x.unsqueeze(1)
        out = self.conv_op(x)
        return out


def BoundaryLoss(prediction, label):
    cost = torch.nn.functional.mse_loss(
        prediction.float(), label.float())
    return torch.sum(cost)
