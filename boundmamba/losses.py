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

class MultiClassDiceLoss(nn.Module):
    def __init__(self, num_classes, smooth=1.0, ignore_index=255):
        super().__init__()
        self.num_classes = num_classes
        self.smooth = smooth
        self.ignore_index = ignore_index

    def forward(self, logits, targets):
        probs = F.softmax(logits, dim=1)
        valid_mask = (targets != self.ignore_index)
        targets_safe = torch.where(valid_mask, targets, torch.zeros_like(targets))
        targets_onehot = F.one_hot(targets_safe, num_classes=self.num_classes).permute(0, 3, 1, 2).float()
        
        probs = probs * valid_mask.unsqueeze(1).float()
        targets_onehot = targets_onehot * valid_mask.unsqueeze(1).float()

        intersection = (probs * targets_onehot).sum(dim=(0, 2, 3))
        union = probs.sum(dim=(0, 2, 3)) + targets_onehot.sum(dim=(0, 2, 3))
        dice = (2.0 * intersection + self.smooth) / (union + self.smooth)
        return 1.0 - dice.mean()

# [SOTA FIX] Binary Focal Loss targets the 72% F1-BCD bottleneck
class BinaryFocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, logits, targets):
        bce_loss = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')
        pt = torch.exp(-bce_loss) # Probability of the correct class
        focal_loss = self.alpha * (1 - pt) ** self.gamma * bce_loss
        return focal_loss.mean()

class SemanticConsistencyLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, logits1, logits2, gt_cd):
        unchanged_mask = (gt_cd == 0).unsqueeze(1).float() 
        probs1 = F.softmax(logits1, dim=1)
        probs2 = F.softmax(logits2, dim=1)
        diff = torch.abs(probs1 - probs2) * unchanged_mask
        num_unchanged = unchanged_mask.sum()
        if num_unchanged > 0:
            return diff.sum() / num_unchanged
        return torch.tensor(0.0, device=logits1.device)

class BoundMambaLoss(nn.Module):
    def __init__(self, num_classes=7, ignore_index=255):
        super().__init__()
        
        self.ce = nn.CrossEntropyLoss(ignore_index=ignore_index)
        self.mc_dice = MultiClassDiceLoss(num_classes=num_classes, ignore_index=ignore_index)
        self.scl = SemanticConsistencyLoss()
        
        # Upgraded to Binary Focal Loss for the Change Mask
        self.bce_focal = BinaryFocalLoss(alpha=0.25, gamma=2.0)
        self.dice = DiceLoss()
        
        self.lambda_ss = 1.0 
        self.lambda_cd = 1.0
        self.lambda_bd = 1.0
        self.lambda_scl = 1.0 

    def forward(self, outputs, targets):
        pred_ss1, pred_ss2, pred_cd, pred_bd = outputs
        gt_ss1, gt_ss2, gt_cd, gt_bd = targets
        
        l_ss = self.ce(pred_ss1, gt_ss1.long()) + self.ce(pred_ss2, gt_ss2.long()) + \
               self.mc_dice(pred_ss1, gt_ss1.long()) + self.mc_dice(pred_ss2, gt_ss2.long())
               
        l_scl = self.scl(pred_ss1, pred_ss2, gt_cd)
        
        # Focal + Dice for BCD
        l_cd = self.bce_focal(pred_cd, gt_cd.float().unsqueeze(1)) + self.dice(torch.sigmoid(pred_cd), gt_cd.float().unsqueeze(1))
        l_bd = self.dice(torch.sigmoid(pred_bd), gt_bd.float().unsqueeze(1))
        
        loss = (self.lambda_ss * l_ss) + (self.lambda_cd * l_cd) + (self.lambda_bd * l_bd) + (self.lambda_scl * l_scl)
        return loss, {"ss": l_ss.item(), "cd": l_cd.item(), "bd": l_bd.item()}
