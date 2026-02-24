import torch
import torch.nn as nn
import torch.nn.functional as F

class DiceLoss(nn.Module):
    def __init__(self, smooth=1.0):
        super().__init__()
        self.smooth = smooth

    def forward(self, pred, target):
        intersection = (pred * target).sum(dim=(2, 3))
        union = pred.sum(dim=(2, 3)) + target.sum(dim=(2, 3))
        dice = (2. * intersection + self.smooth) / (union + self.smooth)
        return 1 - dice.mean()

class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2, ignore_index=255):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.ignore_index = ignore_index

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, ignore_index=self.ignore_index, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        return focal_loss.mean()

class BoundMambaLoss(nn.Module):
    def __init__(self, ignore_index=255):
        super().__init__()
        # Empirically calculated weights for SECOND dataset
        weights = torch.tensor([1.0, 2.5, 1.2, 1.0, 1.2, 2.0, 3.0])
        self.ce = nn.CrossEntropyLoss(weight=weights, ignore_index=ignore_index)
        self.focal = FocalLoss(ignore_index=ignore_index)
        self.bce = nn.BCEWithLogitsLoss()
        self.dice = DiceLoss()
        
        # Heavy emphasis on semantics
        self.lambda_ss = 2.0 
        self.lambda_cd = 1.0
        self.lambda_bd = 1.0

    def forward(self, outputs, targets):
        pred_ss1, pred_ss2, pred_cd, pred_bd = outputs
        gt_ss1, gt_ss2, gt_cd, gt_bd = targets
        
        # Combine CE and Focal for Semantic (OHEM effect)
        l_ss1 = self.ce(pred_ss1, gt_ss1.long()) + self.focal(pred_ss1, gt_ss1.long())
        l_ss2 = self.ce(pred_ss2, gt_ss2.long()) + self.focal(pred_ss2, gt_ss2.long())
        
        l_cd_bce = self.bce(pred_cd, gt_cd.float().unsqueeze(1))
        l_cd_dice = self.dice(torch.sigmoid(pred_cd), gt_cd.float().unsqueeze(1))
        l_cd = l_cd_bce + l_cd_dice
        
        l_bd = self.dice(torch.sigmoid(pred_bd), gt_bd.float().unsqueeze(1))
        
        loss = (self.lambda_ss * (l_ss1 + l_ss2)) + (self.lambda_cd * l_cd) + (self.lambda_bd * l_bd)
        
        return loss, {"ss": (l_ss1+l_ss2).item(), "cd": l_cd.item(), "bd": l_bd.item()}
