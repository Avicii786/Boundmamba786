import torch
import torch.nn as nn
import torch.nn.functional as F

class DiceLoss(nn.Module):
    def __init__(self, smooth=1.0):
        super().__init__()
        self.smooth = smooth

    def forward(self, pred, target):
        # pred: sigmoid [B, 1, H, W]
        # target: binary [B, 1, H, W]
        intersection = (pred * target).sum(dim=(2, 3))
        union = pred.sum(dim=(2, 3)) + target.sum(dim=(2, 3))
        dice = (2. * intersection + self.smooth) / (union + self.smooth)
        return 1 - dice.mean()

class JointLoss(nn.Module):
    def __init__(self, ignore_index=255):
        super().__init__()
        self.ce = nn.CrossEntropyLoss(ignore_index=ignore_index)
        self.bce = nn.BCEWithLogitsLoss()
        self.dice = DiceLoss()
        
    def forward(self, outputs, targets):
        """
        outputs: (ss1, ss2, cd, bd) - all logits
        targets: (gt_ss1, gt_ss2, gt_cd, gt_bd)
        """
        pred_ss1, pred_ss2, pred_cd, pred_bd = outputs
        gt_ss1, gt_ss2, gt_cd, gt_bd = targets
        
        # SS Loss (Cross Entropy)
        l_ss1 = self.ce(pred_ss1, gt_ss1.long())
        l_ss2 = self.ce(pred_ss2, gt_ss2.long())
        
        # CD Loss (BCE + Dice)
        l_cd_bce = self.bce(pred_cd, gt_cd.float().unsqueeze(1))
        l_cd_dice = self.dice(torch.sigmoid(pred_cd), gt_cd.float().unsqueeze(1))
        l_cd = l_cd_bce + l_cd_dice
        
        # Boundary Loss (Dice)
        l_bd = self.dice(torch.sigmoid(pred_bd), gt_bd.float().unsqueeze(1))
        
        # Weighted Sum
        # We weigh boundary higher initially to force structure learning
        loss = l_ss1 + l_ss2 + l_cd + 0.5 * l_bd
        
        return loss, {"ss": (l_ss1+l_ss2).item(), "cd": l_cd.item(), "bd": l_bd.item()}