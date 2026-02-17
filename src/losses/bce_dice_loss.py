from torch import nn
import torch

class DiceBCELoss(nn.Module):
    def __init__(self, smooth=1.0, bce_weight=0.5, dice_weight=0.5):
        super(DiceBCELoss, self).__init__()
        self.bce = nn.BCEWithLogitsLoss()  # includes sigmoid inside
        self.smooth = smooth
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight

    def forward(self, preds, targets):
        # BCE part
        bce_loss = self.bce(preds, targets)

        # Dice part
        preds = torch.sigmoid(preds)  # convert logits -> probabilities
        preds = preds.view(-1)
        targets = targets.view(-1)

        intersection = (preds * targets).sum()
        dice_loss = 1 - (2. * intersection + self.smooth) / (preds.sum() + targets.sum() + self.smooth)

        # combine
        loss = self.bce_weight * bce_loss + self.dice_weight * dice_loss
        return loss