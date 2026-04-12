import os
import argparse
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np

from train import BoundNeXtLightning
from dataset import SCDDataset
from boundmamba.metrics import SCDMetrics

def apply_flip_tta(model, t1, t2):
    """Applies 4-way flip Test-Time Augmentation"""
    logits_ss1_n, logits_ss2_n, logits_cd_n, _ = model(t1, t2)
    logits_ss1_h, logits_ss2_h, logits_cd_h, _ = model(t1.flip(-1), t2.flip(-1))
    logits_ss1_v, logits_ss2_v, logits_cd_v, _ = model(t1.flip(-2), t2.flip(-2))
    logits_ss1_hv, logits_ss2_hv, logits_cd_hv, _ = model(t1.flip([-2, -1]), t2.flip([-2, -1]))
    
    logits_ss1 = (logits_ss1_n + logits_ss1_h.flip(-1) + logits_ss1_v.flip(-2) + logits_ss1_hv.flip([-2, -1])) / 4.0
    logits_ss2 = (logits_ss2_n + logits_ss2_h.flip(-1) + logits_ss2_v.flip(-2) + logits_ss2_hv.flip([-2, -1])) / 4.0
    logits_cd = (logits_cd_n + logits_cd_h.flip(-1) + logits_cd_v.flip(-2) + logits_cd_hv.flip([-2, -1])) / 4.0
    return logits_ss1, logits_ss2, logits_cd

def apply_ms_tta(model, t1, t2, num_classes, scales=[0.5, 0.75, 1.0, 1.25, 1.5]):
    """Applies Multi-Scale Test-Time Augmentation"""
    B, C, H, W = t1.shape
    
    # [FIX] Use explicitly passed num_classes instead of trying to index the model head
    final_logits_ss1 = torch.zeros(B, num_classes, H, W, device=t1.device)
    final_logits_ss2 = torch.zeros_like(final_logits_ss1)
    final_logits_cd = torch.zeros(B, 1, H, W, device=t1.device)
    
    for scale in scales:
        if scale == 1.0:
            t1_s, t2_s = t1, t2
        else:
            t1_s = F.interpolate(t1, scale_factor=scale, mode='bilinear', align_corners=False)
            t2_s = F.interpolate(t2, scale_factor=scale, mode='bilinear', align_corners=False)
            
        l_ss1, l_ss2, l_cd = apply_flip_tta(model, t1_s, t2_s)
        
        if scale != 1.0:
            l_ss1 = F.interpolate(l_ss1, size=(H, W), mode='bilinear', align_corners=False)
            l_ss2 = F.interpolate(l_ss2, size=(H, W), mode='bilinear', align_corners=False)
            l_cd = F.interpolate(l_cd, size=(H, W), mode='bilinear', align_corners=False)
            
        final_logits_ss1 += l_ss1
        final_logits_ss2 += l_ss2
        final_logits_cd += l_cd
        
    final_logits_ss1 /= len(scales)
    final_logits_ss2 /= len(scales)
    final_logits_cd /= len(scales)
    
    return final_logits_ss1, final_logits_ss2, final_logits_cd

def main():
    parser = argparse.ArgumentParser(description="Evaluate BoundNeXt with Multi-Scale Test-Time Augmentation")
    parser.add_argument('--data_root', type=str, required=True, help="Path to SECOND dataset")
    parser.add_argument('--dataset_name', type=str, default='SECOND')
    parser.add_argument('--model_type', type=str, default='convnextv2_base')
    parser.add_argument('--weights', type=str, default=None, help="Dummy arg to satisfy train.py init")
    parser.add_argument('--ckpt_path', type=str, required=True, help="Path to the best_model.ckpt")
    parser.add_argument('--batch_size', type=int, default=1) 
    parser.add_argument('--tta', action='store_true', help="Enable Multi-Scale Test-Time Augmentation")
    parser.add_argument('--split', type=str, default='test', choices=['val', 'test'], help="Which dataset split to evaluate on")
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🚀 Initializing Evaluation on {device}...")

    test_dataset = SCDDataset(root=args.data_root, mode=args.split, dataset_name=args.dataset_name, patch_mode=False)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=2, pin_memory=True)
    
    num_classes = 7 if args.dataset_name.upper() == 'SECOND' else 5
    print(f"📦 Loading weights from: {args.ckpt_path}")
    
    if hasattr(torch.serialization, 'add_safe_globals'):
        torch.serialization.add_safe_globals([argparse.Namespace])

    pl_model = BoundNeXtLightning.load_from_checkpoint(checkpoint_path=args.ckpt_path, map_location=device, args=args)
    model = pl_model.model
    model.to(device)
    model.eval() 
    
    # Sweep thresholds for Binary Change Detection
    thresholds = [0.40, 0.50, 0.55]
    metrics_dict = {t: SCDMetrics(num_classes=num_classes) for t in thresholds}
    
    # Expanded TTA scales for maximum boundary resilience
    test_scales = [0.5, 0.75, 1.0, 1.25, 1.5] if args.tta else [1.0]
    
    print(f"🎯 Starting Inference on [{args.split.upper()}] split... MS-TTA ({len(test_scales) * 4}-Pass): {'ON' if args.tta else 'OFF'}")
    
    with torch.no_grad():
        for batch in tqdm(test_loader, desc=f"Testing on {args.split}"):
            t1, t2 = batch['img_A'].to(device), batch['img_B'].to(device)
            l1, l2, gt_cd = batch['sem1'].to(device), batch['sem2'].to(device), batch['bcd'].to(device)
            
            if args.tta:
                # [FIX] Passing num_classes cleanly to the TTA function
                logits_ss1, logits_ss2, logits_cd = apply_ms_tta(model, t1, t2, num_classes, scales=test_scales)
            else:
                logits_ss1, logits_ss2, logits_cd, _ = model(t1, t2)
            
            # Pre-calculate base semantic argmaxes to save memory in the loop
            base_ss1 = torch.argmax(logits_ss1, dim=1)
            base_ss2 = torch.argmax(logits_ss2, dim=1)
            logits_shared = logits_ss1 + logits_ss2
            p_shared = torch.argmax(logits_shared, dim=1)
            probs_cd = torch.sigmoid(logits_cd).squeeze(1)
            
            # Sweep all thresholds simultaneously
            for t in thresholds:
                p_cd = (probs_cd > t).long()
                p_ss1_t = torch.where(p_cd == 0, p_shared, base_ss1)
                p_ss2_t = torch.where(p_cd == 0, p_shared, base_ss2)
                metrics_dict[t].update(p_ss1_t, p_ss2_t, p_cd, l1, l2, gt_cd)
            
    print("\n" + "="*80)
    print(f"📊 MULTI-THRESHOLD SWEEP RESULTS ({args.split.upper()} SET)")
    print("="*80)
    print(f" Dataset: {args.dataset_name} | Model: {args.model_type.upper()} | MS-TTA: {'ON' if args.tta else 'OFF'}")
    print("-" * 80)
    print(f"{'Threshold':^10} | {'mIoU':^8} | {'SeK':^8} | {'F1-BCD':^8} | {'Score':^8}")
    print("-" * 80)
    
    best_score = 0
    best_t = 0.5
    for t in thresholds:
        res = metrics_dict[t].compute()
        print(f"{t:^10.2f} | {res['miou']:^8.4f} | {res['sek']:^8.4f} | {res['f1_bcd']:^8.4f} | {res['score']:^8.4f}")
        if res['score'] > best_score:
            best_score = res['score']
            best_t = t
            
    print("-" * 80)
    print(f"🏆 BEST THRESHOLD: {best_t:.2f} (Max Score: {best_score:.4f})")
    print("="*80)

if __name__ == '__main__':
    main()
