import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn


class MixSoftmaxCrossEntropyLoss(nn.CrossEntropyLoss):
    def __init__(self, weight=None, aux=False, aux_weight=0.2, ignore_index=-1):
        super(MixSoftmaxCrossEntropyLoss, self).__init__(ignore_index=ignore_index)
        self.aux = aux
        self.aux_weight = aux_weight

    def _aux_forward(self, *inputs):
        *preds, target = tuple(inputs)

        loss = super(MixSoftmaxCrossEntropyLoss, self).forward(preds[0], target)
        for i in range(1, len(preds)):
            aux_loss = super(MixSoftmaxCrossEntropyLoss, self).forward(preds[i], target)
            loss += self.aux_weight * aux_loss
        return loss

    def forward(self, preds, target):
        inputs = tuple([preds, target.long()])
        if self.aux:
            loss_dict = dict(loss=self._aux_forward(*inputs))
            losses = sum(loss for loss in loss_dict.values())
            return losses
        else:
            loss_dict =dict(loss=super(MixSoftmaxCrossEntropyLoss, self).forward(*inputs))
            losses = sum(loss for loss in loss_dict.values())
            return losses

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
            prediction.float(),label.float())
    return torch.sum(cost)


def make_one_hot(labels, classes):
    labels = labels.unsqueeze(1)
    one_hot = torch.FloatTensor(labels.size()[0], classes, labels.size()[2], labels.size()[3]).zero_().to(labels.device)
    target = one_hot.scatter_(1, labels.data.long(), 1)
    return target


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

    def forward(self, predict, target):
        assert predict.shape[0] == target.shape[0], "predict & target batch size don't match"
        predict = predict.contiguous().view(predict.shape[0], -1)
        target = target.contiguous().view(target.shape[0], -1)

        num = torch.sum(torch.mul(predict, target), dim=1) + self.smooth
        den = torch.sum(predict.pow(self.p) + target.pow(self.p), dim=1) + self.smooth

        loss = 1 - num / den

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        elif self.reduction == 'none':
            return loss
        else:
            raise Exception('Unexpected reduction {}'.format(self.reduction))


class DiceLoss(nn.Module):
    """Dice loss, need one hot encode input
    Args:
        weight: An array of shape [num_classes,]
        ignore_index: class index to ignore
        predict: A tensor of shape [N, C, *]
        target: A tensor of same shape with predict
        other args pass to BinaryDiceLoss
    Return:
        same as BinaryDiceLoss
    """
    def __init__(self, weight=None, ignore_index=None, **kwargs):
        super(DiceLoss, self).__init__()
        self.kwargs = kwargs
        self.weight = weight
        self.ignore_index = ignore_index

    def forward(self, predict, target):
        target = make_one_hot(target, predict.size()[1])
        assert predict.shape == target.shape, 'predict & target shape do not match'
        dice = BinaryDiceLoss(**self.kwargs)
        total_loss = 0
        predict = F.softmax(predict, dim=1)

        for i in range(target.shape[1]):
            if i != self.ignore_index:
                dice_loss = dice(predict[:, i], target[:, i])
                if self.weight is not None:
                    assert self.weight.shape[0] == target.shape[1], \
                        'Expect weight shape [{}], get[{}]'.format(target.shape[1], self.weight.shape[0])
                    dice_loss *= self.weights[i]
                total_loss += dice_loss

        return total_loss/target.shape[1]


def get_weights(target):
    t_np = target.view(-1).data.cpu().numpy()

    classes, counts = np.unique(t_np, return_counts=True)
    cls_w = np.median(counts) / counts
    # cls_w = class_weight.compute_class_weight('balanced', classes, t_np)

    weights = np.ones(7)
    weights[classes] = cls_w
    return torch.from_numpy(weights).float().cuda()


class CrossEntropyLoss2d(nn.Module):
    def __init__(self, weight=None, ignore_index=255, reduction='mean'):
        super(CrossEntropyLoss2d, self).__init__()
        self.CE = nn.CrossEntropyLoss(weight=weight, ignore_index=ignore_index, reduction=reduction)

    def forward(self, output, target):
        # output = F.softmax(output, dim=1)
        loss = self.CE(output, target.long())
        return loss


class WeightedCrossEntropyLoss(torch.nn.CrossEntropyLoss):
   """
   Network has to have NO NONLINEARITY!
   """
   def __init__(self, weight=None):
       super(WeightedCrossEntropyLoss, self).__init__()
       self.weight = weight

   def forward(self, inp, target):
       target = target.long()
       num_classes = inp.size()[1]

       i0 = 1
       i1 = 2

       while i1 < len(inp.shape): # this is ugly but torch only allows to transpose two axes at once
           inp = inp.transpose(i0, i1)
           i0 += 1
           i1 += 1

       inp = inp.contiguous()
       inp = inp.view(-1, num_classes)

       target = target.view(-1,)
       wce_loss = torch.nn.CrossEntropyLoss(weight=self.weight)

       return wce_loss(inp, target)


class BCELoss(nn.Module):
    def __init__(self, weight=None, smooth=1., ignore_index=255):
        super(BCELoss, self).__init__()
        self.ignore_index = ignore_index
        self.smooth = smooth
        self.BCE = nn.BCELoss()

    def forward(self, output, target):
        output = output.squeeze(1)
        loss = self.BCE(output, target.float())

        return loss


class BCE_BDLoss(nn.Module):
    def __init__(self, weight=None, loss_weight=None):
        super(BCE_BDLoss, self).__init__()
        self.BCE = nn.BCELoss()
        self.loss_weight = loss_weight
        self.edge_conv = EdgeConv().cuda()

    def forward(self, output, target):
        output = output.squeeze(1)
        loss = self.BCE(output, target.float())
        output_bd = self.edge_conv(output)
        target_bd = self.edge_conv(target.float())
        loss_bd = BoundaryLoss(output_bd, target_bd)
        return loss*self.loss_weight[0]+loss_bd*self.loss_weight[1]


class FocalLoss2d(nn.Module):

    def __init__(self, weight=None, gamma=0, size_average=True):
        super(FocalLoss2d, self).__init__()

        self.gamma = gamma
        self.weight = weight
        self.size_average = size_average

    def forward(self, input, target):
        target = target.long()
        if input.dim()>2:
            input = input.contiguous().view(input.size(0), input.size(1), -1)
            input = input.transpose(1,2)
            input = input.contiguous().view(-1, input.size(2)).squeeze()
        if target.dim()==4:
            target = target.contiguous().view(target.size(0), target.size(1), -1)
            target = target.transpose(1,2)
            target = target.contiguous().view(-1, target.size(2)).squeeze()
        elif target.dim()==3:
            target = target.view(-1)
        else:
            target = target.view(-1, 1)

        # compute the negative likelyhood
        logpt = -F.cross_entropy(input, target)
        pt = torch.exp(logpt)

        # compute the loss
        loss = -((1-pt)**self.gamma) * logpt

        # averaging (or not) loss
        if self.size_average:
            return loss.mean()
        else:
            return loss.sum()


class CE_DiceLoss(nn.Module):
    def __init__(self, weight=None, smooth=1, reduction='mean', ignore_index=255):
        super(CE_DiceLoss, self).__init__()
        self.smooth = smooth
        self.dice = DiceLoss()
        self.cross_entropy = nn.CrossEntropyLoss(weight=weight, reduction=reduction, ignore_index=ignore_index)

    def forward(self, output, target):
        CE_loss = self.cross_entropy(output, target)
        dice_loss = self.dice(output, target)
        return CE_loss + dice_loss


class TriLoss(nn.Module):
    def __init__(self, weight=None, reduction='mean'):
        super(TriLoss, self).__init__()
        self.dice = DiceLoss()
        weights_f = torch.tensor([0.01, 1]).cuda().float()
        weights_b = torch.tensor([1, 0.01]).cuda().float()
        self.cef = nn.CrossEntropyLoss(weight=weights_f, reduction=reduction)
        self.ceb = nn.CrossEntropyLoss(weight=weights_b, reduction=reduction)
        self.BCE = nn.BCELoss()

    def forward(self, output, target):
        out, out_b, out_f = output
        out = out.squeeze(1)
        losses1 = self.BCE(out, target.float())  # calculate loss
        losses_f = self.cef(out_f, target)  # calculate loss
        losses_b = self.ceb(out_b, target)  # calculate loss
        losses = losses1 + losses_b * 0.3 + losses_f * 0.7
        return losses

def lovasz_grad(gt_sorted):
    """
    Computes gradient of the Lovasz extension w.r.t sorted errors
    See Alg. 1 in paper
    """
    p = len(gt_sorted)
    gts = gt_sorted.sum()
    intersection = gts - gt_sorted.float().cumsum(0)
    union = gts + (1 - gt_sorted).float().cumsum(0)
    jaccard = 1. - intersection / union
    if p > 1:  # cover 1-pixel case
        jaccard[1:p] = jaccard[1:p] - jaccard[0:-1]
    return jaccard


class LovaszSoftmax(nn.Module):
    def __init__(self, weight=None, reduction='mean'):
        super(LovaszSoftmax, self).__init__()
        self.reduction = reduction

    def prob_flatten(self, input, target):
        assert input.dim() in [4, 5]
        num_class = input.size(1)
        input = input.permute(0, 2, 3, 1).contiguous()
        input_flatten = input.view(-1, num_class)
        target_flatten = target.view(-1)
        return input_flatten, target_flatten

    def lovasz_softmax_flat(self, inputs, targets):
        num_classes = inputs.size(1)
        losses = []
        for c in range(num_classes):
            target_c = (targets == c).float()
            if num_classes == 1:
                input_c = inputs[:, 0]
            else:
                input_c = inputs[:, c]
            loss_c = (torch.autograd.Variable(target_c) - input_c).abs()
            loss_c_sorted, loss_index = torch.sort(loss_c, 0, descending=True)
            target_c_sorted = target_c[loss_index]
            losses.append(torch.dot(loss_c_sorted, torch.autograd.Variable(lovasz_grad(target_c_sorted))))
        losses = torch.stack(losses)

        if self.reduction == 'none':
            loss = losses
        elif self.reduction == 'sum':
            loss = losses.sum()
        else:
            loss = losses.mean()
        return loss

    def forward(self, inputs, targets):
        # print(inputs.shape, targets.shape) # (batch size, class_num, x,y,z), (batch size, 1, x,y,z)
        inputs, targets = self.prob_flatten(inputs, targets)
        # print(inputs.shape, targets.shape)
        losses = self.lovasz_softmax_flat(inputs, targets)
        return losses


class net(nn.Module):
    def __init__(self, in_channels, num_classes):
        super(net, self).__init__()
        self.conv = nn.Conv2d(in_channels, num_classes, (3, 3), padding=1)

    def forward(self, input):
        out = self.conv(input)
        return out


# def get_loss(loss_type, outputs, targets, class_weights=None):
#     if loss_type == 'CrossEntropyLoss2d':
#         loss_func = CrossEntropyLoss2d(weight=class_weights)
#         loss = loss_func(outputs, targets.long())
#         return loss
#     if loss_type == 'WeightedCrossEntropyLoss':
#         loss_func = WeightedCrossEntropyLoss(weight=class_weights)
#         loss = loss_func(outputs, targets.long())
#         return loss
#     elif loss_type == 'DiceLoss':
#         loss_func = DiceLoss(weight=class_weights)
#         loss = loss_func(outputs, targets)
#         return loss
#     elif loss_type == 'BinaryDiceLoss':
#         loss_func = BinaryDiceLoss()
#         loss = loss_func(outputs, targets)
#         return loss
#     elif loss_type == 'FocalLoss2d':
#         loss_func = FocalLoss2d()
#         loss = loss_func(outputs, targets.float())
#         return loss
#     elif loss_type == 'CE_DiceLoss':
#         loss_func = CE_DiceLoss()
#         loss = loss_func(outputs, targets.long())
#         return loss
#     elif loss_type == 'LovaszSoftmax':
#         loss_func = LovaszSoftmax()
#         loss = loss_func(outputs, targets.long())
#         return loss
#     elif loss_type == 'bce':
#         loss_func = BCELoss()
#         loss = loss_func(outputs, targets)
#         return loss
#     elif loss_type == 'bce_bd':
#         loss_func = BCE_BDLoss([0.7, 0.3])
#         loss = loss_func(outputs, targets)
#         return loss
#     elif loss_type == 'triple':
#         loss_func = BCE_BDLoss([0.7, 0.3])
#         loss = loss_func(outputs, targets)
#         return loss
#     else:
#         return None


if __name__ == '__main__':
    from torch.optim import Adam

    data = torch.rand(2, 3, 64, 64)
    model = net(3, 8)
    target = torch.zeros(2, 64, 64).random_(8)
    Loss = LovaszSoftmax()
    optim = Adam(model.parameters(), lr=0.01, betas=(0.99, 0.999))
    for step in range(10):
        out = model(data)
        loss = Loss(out, target)
        optim.zero_grad()
        loss.backward()
        optim.step()
        print(loss)