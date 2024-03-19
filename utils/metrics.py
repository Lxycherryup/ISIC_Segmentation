import torch
import torch.nn as nn
import torch.nn.functional as F

""" Loss Functions -------------------------------------- """
class DiceLoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(DiceLoss, self).__init__()

    def forward(self, inputs, targets, smooth=1):
        inputs = torch.sigmoid(inputs)

        inputs = inputs.view(-1)

        targets = targets.view(-1)

        intersection = (inputs * targets).sum()
        dice = (2.*intersection + smooth)/(inputs.sum() + targets.sum() + smooth)

        return 1 - dice




class StructureLoss(nn.Module):
    def __init__(self):
        super(StructureLoss, self).__init__()

    def forward(self, pred, mask):
        # 计算加权因子，强调边缘或变化区域
        # 例如，使用边缘区域的局部平均差异作为加权因子
        weit = 1 + 5 * torch.abs(F.avg_pool2d(mask, kernel_size=31, stride=1, padding=15) - mask)

        # 计算加权二值交叉熵损失
        pred_flat = pred.view(-1)
        mask_flat = mask.view(-1)
        weit_flat = weit.view(-1)
        wbce = F.binary_cross_entropy_with_logits(pred_flat, mask_flat, reduction='none')
        wbce = (weit_flat * wbce).sum() / weit_flat.sum()

        # 使用sigmoid激活函数处理预测值
        pred = torch.sigmoid(pred)

        # 计算加权交集和并集
        intersection = (pred * mask * weit).sum()
        union = (pred + mask) * weit.sum()

        # 计算加权Dice损失
        dice_loss = 1 - (2. * intersection + 1) / (union - intersection + 1)

        # 结合加权BCE和Dice损失
        structure_loss = wbce + dice_loss

        return (wbce+dice_loss).mean()

""" Metrics ------------------------------------------ """
def precision(y_true, y_pred):
    intersection = (y_true * y_pred).sum()
    return (intersection + 1e-15) / (y_pred.sum() + 1e-15)

def recall(y_true, y_pred):
    intersection = (y_true * y_pred).sum()
    return (intersection + 1e-15) / (y_true.sum() + 1e-15)

def F2(y_true, y_pred, beta=2):
    p = precision(y_true,y_pred)
    r = recall(y_true, y_pred)
    return (1+beta**2.) *(p*r) / float(beta**2*p + r + 1e-15)

def dice_score(y_true, y_pred):
    return (2 * (y_true * y_pred).sum() + 1e-15) / (y_true.sum() + y_pred.sum() + 1e-15)

def jac_score(y_true, y_pred):
    intersection = (y_true * y_pred).sum()
    union = y_true.sum() + y_pred.sum() - intersection
    return (intersection + 1e-15) / (union + 1e-15)
