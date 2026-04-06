import torch
import torch.nn.functional as F
import numpy as np

def _get_categorical_boundary(mask, kernel_size=5):
    """
    Uses Max/Min Pooling as morphological Dilation/Erosion.
    This safely extracts structural edges from BOTH binary and multi-class maps
    without doing meaningless arithmetic on categorical labels.
    """
    if mask.dim() == 3:
        mask = mask.unsqueeze(1) # [B, 1, H, W]
    
    mask = mask.float()
    padding = kernel_size // 2
    
    # Max-pool acts as morphological dilation
    dilation = F.max_pool2d(mask, kernel_size, stride=1, padding=padding)
    
    # Min-pool acts as morphological erosion (achieved via negating the mask)
    erosion = -F.max_pool2d(-mask, kernel_size, stride=1, padding=padding)
    
    # Where max != min, there is a semantic class transition (a boundary)
    boundary = (dilation != erosion).float()
    return boundary.squeeze(1)

def extract_composite_boundary(gt_cd, sem1, sem2, kernel_size=5):
    """
    [SOTA FIX 5] Composite Boundary Extraction.
    Extracts thick (5x5) boundaries not just from the change mask, but also from the 
    semantic structures of T1 and T2. This forces the Boundary Head to learn 
    the intra-image edges (e.g., building vs. road) even if they didn't change.
    """
    bd_cd = _get_categorical_boundary(gt_cd, kernel_size)
    bd_sem1 = _get_categorical_boundary(sem1, kernel_size)
    bd_sem2 = _get_categorical_boundary(sem2, kernel_size)
    
    # Logical OR: A pixel is a boundary if it's a change edge, T1 edge, OR T2 edge.
    composite_bd = torch.clamp(bd_cd + bd_sem1 + bd_sem2, 0.0, 1.0)
    
    return composite_bd

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
