import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


class FocalLoss(nn.Module):
    def __init__(self, gamma=0, alpha=None, size_average=True):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        if isinstance(alpha, (float, int)): self.alpha = torch.Tensor([alpha, 1 - alpha])
        if isinstance(alpha, list): self.alpha = torch.Tensor(alpha)
        self.size_average = size_average

    def forward(self, input, target):
        if input.dim() > 2:
            input = input.view(input.size(0), input.size(1), -1)  # N,C,H,W => N,C,H*W
            input = input.transpose(1, 2)  # N,C,H*W => N,H*W,C
            input = input.contiguous().view(-1, input.size(2))  # N,H*W,C => N*H*W,C
        target = target.view(-1, 1)

        logpt = F.log_softmax(input, dim=1)
        logpt = logpt.gather(1, target)
        logpt = logpt.view(-1)
        pt = Variable(logpt.data.exp())

        if self.alpha is not None:
            if self.alpha.type() != input.data.type():
                self.alpha = self.alpha.type_as(input.data)
            at = self.alpha.gather(0, target.data.view(-1))
            logpt = logpt * Variable(at)

        loss = -1 * (1 - pt) ** self.gamma * logpt
        if self.size_average:
            return loss.mean()
        else:
            return loss.sum()


class SoftDiceLoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(SoftDiceLoss, self).__init__()

    def forward(self, logits, targets, epsilon=1e-6):
        num_classes = logits.shape[1]
        dice = 0.0

        for i in range(num_classes):
            predicted = torch.softmax(logits[:, i, :, :], dim=1)  # Apply softmax along the channel dimension
            target = (targets == i).float()
            dice += self.dice_coefficient(predicted, target, epsilon=epsilon)

        loss = dice / num_classes
        return loss

    def dice_coefficient(self, predicted, target, epsilon=1e-5):
        intersection = torch.sum(predicted * target)
        union = torch.sum(predicted) + torch.sum(target)
        dice = 1 - (2.0 * intersection + epsilon) / (union + epsilon)
        return dice


class MaskIoULoss(nn.Module):

    def __init__(self, ):
        super(MaskIoULoss, self).__init__()

    def forward(self, logits, ground_truth_mask, pred_iou):
        """
        pred_mask: [B, 1, H, W]
        ground_truth_mask: [B, 1, H, W]
        pred_iou: [B, 1]
        """
        pred_mask = torch.argmax(logits, dim=1)
        assert pred_mask.shape == ground_truth_mask.shape, "pred_mask and ground_truth_mask should have the same shape."

        p = torch.sigmoid(pred_mask)
        intersection = torch.sum(p * ground_truth_mask)
        union = torch.sum(p) + torch.sum(ground_truth_mask) - intersection
        iou = (intersection + 1e-7) / (union + 1e-7)
        iou_loss = torch.mean((iou - pred_iou) ** 2)
        return iou_loss


class FDLoss(nn.Module):
    def __init__(self, weight=2.0, gamma=2.0):
        super(FDLoss, self).__init__()
        self.weight = weight
        self.focal_loss = FocalLoss(gamma=gamma)
        self.dice_loss = SoftDiceLoss()

    def forward(self, logits, ground_truth_mask):
        focal_loss = self.focal_loss(logits, ground_truth_mask)
        dice_loss = self.dice_loss(logits, ground_truth_mask)
        return self.weight * focal_loss + dice_loss


class CFDLoss(nn.Module):
    def __init__(self):
        super(CFDLoss, self).__init__()
        self.ce_loss = nn.CrossEntropyLoss()
        self.fd_loss = FDLoss()
        self.step = 0

    def forward(self, logits, ground_truth_mask):
        ce_loss = self.ce_loss(logits, ground_truth_mask)
        fd_loss = self.fd_loss(logits, ground_truth_mask)

        return ce_loss + fd_loss

        # return ce_loss * math.exp(-epoch / 100) + fd_loss * (1 - math.exp(-epoch / 100))


class FDILoss(nn.Module):
    def __init__(self, weight=2.0, iou_scale=1.0):
        super(FDILoss, self).__init__()
        self.weight = weight
        self.iou_scale = iou_scale
        self.focal_loss = FocalLoss()
        self.dice_loss = SoftDiceLoss()
        self.iou_loss = MaskIoULoss()

    def forward(self, logits, ground_truth_mask, pred_iou):
        focal_loss = self.focal_loss(logits, ground_truth_mask)
        dice_loss = self.dice_loss(logits, ground_truth_mask)
        iou_loss = self.iou_loss(logits, ground_truth_mask, pred_iou)
        return self.weight * focal_loss + dice_loss + self.iou_scale * iou_loss


if __name__ == '__main__':
    data = torch.randn(2, 6, 4, 4)
    label = torch.randint(0, 6, (2, 4, 4))
    iou = torch.randn(2, 1)
    for epoch in range(500):
        print(f'{epoch}:', math.exp(-epoch / 100))
