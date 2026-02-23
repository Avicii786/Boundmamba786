import torch
import torch.nn.functional as F
import numpy as np

def extract_boundary(mask):
    """
    Extracts binary boundary from semantic mask using morphological gradient.
    mask: [B, H, W]
    Returns: [B, H, W] (Float 0/1)
    """
    if mask.dim() == 3:
        mask = mask.unsqueeze(1) # [B, 1, H, W]
    
    mask = mask.float()
    kernel = torch.ones(1, 1, 3, 3, device=mask.device)
    
    # Dilation
    dilation = (F.conv2d(mask, kernel, padding=1) > 0).float()
    
    # Erosion (using inverse dilation)
    erosion = 1.0 - (F.conv2d(1.0 - mask, kernel, padding=1) > 0).float()
    
    # Boundary = Dilation - Erosion
    boundary = dilation - erosion
    return boundary.squeeze(1)

def calculate_metrics(pred_cd, gt_cd):
    """
    pred_cd: Logits [B, 1, H, W]
    gt_cd: Mask [B, H, W]
    """
    probs = torch.sigmoid(pred_cd).squeeze(1)
    preds = (probs > 0.5).long()
    gt = gt_cd.long()
    
    tp = (preds * gt).sum().item()
    fp = (preds * (1 - gt)).sum().item()
    fn = ((1 - preds) * gt).sum().item()
    
    precision = tp / (fp + tp + 1e-6)
    recall = tp / (fn + tp + 1e-6)
    f1 = 2 * precision * recall / (precision + recall + 1e-6)
    iou = tp / (fp + fn + tp + 1e-6)
    
    return iou, f1