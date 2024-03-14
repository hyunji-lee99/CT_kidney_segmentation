import torch.nn.functional as F
import torch
import numpy as np

def DiceLoss(predicted, target, smooth=1e-10):
    predicted = F.softmax(predicted, dim=1)
    # To address data imbalance,remove background class
    target = target[:, 1:]
    predicted = predicted[:, 1:]
    # if (p>=0.5) 1.0 else 0.0
    torch.where(predicted>=0.5, 1.0, 0.0)

    # dim 추가해서 각 클래스에 대해서 dice 값 계산하고
    intersection = torch.sum(target * predicted, dim=(2, 3))
    gt = torch.sum(target, dim=(2, 3))
    pred = torch.sum(predicted, dim=(2, 3))
    total = gt + pred
    union = total - intersection
    # print(f"intersection : {intersection}, total:{total}, union:{union}")

    # dice coefficient
    dice = (2. * intersection + smooth) / (total + smooth)
    # 출력되는 두 개의 클래스에 대한 dice를 .mean()내서 반영
    dice_loss = (1. - dice).mean()
    # print(f"dice loss : {dice_loss}")

    IoU = ((intersection + smooth) / (union + smooth)).mean()
    # print(f"iou : {IoU}")

    return dice_loss, IoU