import torch
import torch.nn.functional as F
import numpy as np

def extract_boundary(mask, kernel_size=5):
    """
    [SOTA FIX 1] Boundary Thickening.
    Increased kernel from 3x3 to 5x5 to create a 'boundary ribbon'.
    A 1-pixel thick boundary is too sparse for Dice Loss to optimize effectively.
    A thicker boundary gives the model a softer target to guide the BCTF attention.
    """
    if mask.dim() == 3:
        mask = mask.unsqueeze(1) # [B, 1, H, W]
    
    mask = mask.float()
    kernel = torch.ones(1, 1, kernel_size, kernel_size, device=mask.device)
    padding = kernel_size // 2
    
    # Dilation
    dilation = (F.conv2d(mask, kernel, padding=padding) > 0).float()
    
    # Erosion (using inverse dilation)
    erosion = 1.0 - (F.conv2d(1.0 - mask, kernel, padding=padding) > 0).float()
    
    # Boundary = Dilation - Erosion
    boundary = dilation - erosion
    return boundary.squeeze(1)

def calculate_metrics(pred_cd, gt_cd):
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
