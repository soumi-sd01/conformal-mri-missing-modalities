import torch
import torch.nn as nn
import torch.nn.functional as F


class DiceCELoss(nn.Module):
    """Foreground multiclass Dice + cross entropy; absent target classes are excluded from Dice."""

    def __init__(self, dice_weight=0.5, ce_weight=0.5, smooth=1e-5):
        super().__init__()
        self.dice_weight = float(dice_weight)
        self.ce_weight = float(ce_weight)
        self.smooth = float(smooth)

    def forward(self, logits, targets):
        targets = targets.long()
        ce = F.cross_entropy(logits.float(), targets)
        probs = F.softmax(logits.float(), dim=1)
        one_hot = F.one_hot(targets, num_classes=logits.shape[1]).permute(0, 4, 1, 2, 3).float()
        probs = probs[:, 1:]
        one_hot = one_hot[:, 1:]
        dims = (0, 2, 3, 4)
        intersection = (probs * one_hot).sum(dim=dims)
        denominator = (probs + one_hot).sum(dim=dims)
        target_volume = one_hot.sum(dim=dims)
        present = target_volume > 0
        if present.any():
            dice = (2 * intersection + self.smooth) / (denominator + self.smooth)
            dice_loss = 1.0 - dice[present].mean()
        else:
            dice_loss = logits.sum() * 0.0
        return self.dice_weight * dice_loss + self.ce_weight * ce
