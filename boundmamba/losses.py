import torch
import torch.nn as nn
import torch.nn.functional as F

class OHEMCrossEntropyLoss(nn.Module):
    """
    [RELAXED FIX] Increased keep_ratio from 0.7 to 0.9.
    When combined with CutMix, throwing away 30% of gradients starves the model.
    Now we only ignore the absolute easiest 10% of pixels.
    """
    def __init__(self, ignore_index=255, keep_ratio=0.9):
        super().__init__()
        self.ignore_index = ignore_index
        self.keep_ratio = keep_ratio
        self.ce = nn.CrossEntropyLoss(ignore_index=ignore_index, reduction='none')

    def forward(self, logits, targets):
        loss = self.ce(logits, targets)
        loss = loss.view(-1)
        valid_mask = (targets.view(-1) != self.ignore_index)
        loss = loss[valid_mask]
        
        if len(loss) == 0:
            return loss.sum()
            
        num_kept = int(len(loss) * self.keep_ratio)
        if num_kept > 0:
            loss, _ = torch.topk(loss, num_kept)
            
        return loss.mean()

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

class BinaryFocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, logits, targets):
        bce_loss = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')
        pt = torch.exp(-bce_loss) 
        focal_loss = self.alpha * (1 - pt) ** self.gamma * bce_loss
        return focal_loss.mean()

class JSDivergenceSCLoss(nn.Module):
    def __init__(self, eps=1e-8):
        super().__init__()
        self.eps = eps

    def forward(self, logits1, logits2, gt_cd):
        unchanged_mask = (gt_cd == 0).unsqueeze(1).float() 
        
        p1 = F.softmax(logits1, dim=1)
        p2 = F.softmax(logits2, dim=1)
        
        m = 0.5 * (p1 + p2)
        
        kl1 = p1 * (torch.log(p1 + self.eps) - torch.log(m + self.eps))
        kl2 = p2 * (torch.log(p2 + self.eps) - torch.log(m + self.eps))
        
        js = 0.5 * (kl1 + kl2)
        js_masked = js.sum(dim=1, keepdim=True) * unchanged_mask
        
        num_unchanged = unchanged_mask.sum()
        if num_unchanged > 0:
            return js_masked.sum() / (num_unchanged + self.eps)
        return torch.tensor(0.0, device=logits1.device)

class BoundMambaLoss(nn.Module):
    def __init__(self, num_classes=7, ignore_index=255):
        super().__init__()
        
        self.ohem_ce = OHEMCrossEntropyLoss(ignore_index=ignore_index, keep_ratio=0.9) # Relaxed
        self.mc_dice = MultiClassDiceLoss(num_classes=num_classes, ignore_index=ignore_index)
        self.scl = JSDivergenceSCLoss()
        
        self.bce_focal = BinaryFocalLoss(alpha=0.25, gamma=2.0)
        self.dice = DiceLoss()
        
        self.lambda_ss = 1.0 
        self.lambda_cd = 2.0  
        self.lambda_bd = 0.5  
        self.lambda_scl = 0.25 # [RELAXED FIX] Dropped from 0.5. Let the network make bolder predictions.

    def forward(self, outputs, targets):
        pred_ss1, pred_ss2, pred_cd, pred_bd = outputs
        gt_ss1, gt_ss2, gt_cd, gt_bd = targets
        
        l_ss = self.ohem_ce(pred_ss1, gt_ss1.long()) + self.ohem_ce(pred_ss2, gt_ss2.long()) + \
               self.mc_dice(pred_ss1, gt_ss1.long()) + self.mc_dice(pred_ss2, gt_ss2.long())
               
        l_scl = self.scl(pred_ss1, pred_ss2, gt_cd)
        
        l_cd = self.bce_focal(pred_cd, gt_cd.float().unsqueeze(1)) + self.dice(torch.sigmoid(pred_cd), gt_cd.float().unsqueeze(1))
        l_bd = self.dice(torch.sigmoid(pred_bd), gt_bd.float().unsqueeze(1))
        
        loss = (self.lambda_ss * l_ss) + (self.lambda_cd * l_cd) + (self.lambda_bd * l_bd) + (self.lambda_scl * l_scl)
        return loss, {"ss": l_ss.item(), "cd": l_cd.item(), "bd": l_bd.item()}
