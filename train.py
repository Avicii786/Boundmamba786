import os
import argparse
import random
import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

# [UPDATED] Use modern PyTorch 2.x AMP imports
from torch.amp import autocast, GradScaler

# Imports from your package
from boundmamba import BoundNeXt, BoundMambaLoss
from boundmamba.utils import extract_boundary
from boundmamba.metrics import SCDMetrics
from dataset import SCDDataset

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print(f"[Setup] Random Seed securely set to: {seed}")

def parse_args():
    parser = argparse.ArgumentParser(description="Train BoundNeXt for Semantic Change Detection")
    parser.add_argument('--data_root', type=str, required=True, help='Path to dataset root')
    parser.add_argument('--dataset_name', type=str, default='SECOND', choices=['SECOND', 'LandsatSCD'])
    parser.add_argument('--model_type', type=str, default='convnextv2_tiny', 
                        choices=['convnextv2_tiny', 'convnextv2_small', 'convnextv2_base', 'convnextv2_large'])
    parser.add_argument('--weights', type=str, default=None)
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--lr', type=float, default=0.0003)
    parser.add_argument('--save_dir', type=str, default='/kaggle/working/checkpoints')
    parser.add_argument('--seed', type=int, default=42)
    return parser.parse_args()

def train(args):
    set_seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    os.makedirs(args.save_dir, exist_ok=True)

    print("\n" + "="*50)
    print("TRAINING CONFIGURATION:")
    for arg, value in vars(args).items():
        print(f"  --{arg}: {value}")
    print("="*50 + "\n")

    num_classes = 7 if args.dataset_name.upper() == 'SECOND' else 5
    
    print(f"Initializing BoundNeXt (Num Classes: {num_classes}, Backbone: {args.model_type})...")
    model = BoundNeXt(
        num_classes=num_classes, 
        pretrained_path=args.weights, 
        model_type=args.model_type    
    ).to(device)
    
    criterion = BoundMambaLoss().to(device)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    
    # [UPDATED] Initialize the modern GradScaler specifying 'cuda'
    scaler = GradScaler('cuda', enabled=torch.cuda.is_available())
    
    metrics = SCDMetrics(num_classes=num_classes)

    print(f"Loading {args.dataset_name} dataset with 512x512 resolution...")
    train_set = SCDDataset(root=args.data_root, mode='train', dataset_name=args.dataset_name, patch_mode=False)
    val_set = SCDDataset(root=args.data_root, mode='val', dataset_name=args.dataset_name, patch_mode=False)
    
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=0, pin_memory=True)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, num_workers=0, pin_memory=True)

    print(f"Train samples: {len(train_set)} | Val samples: {len(val_set)}")

    best_score = 0.0

    for epoch in range(args.epochs):
        model.train()
        epoch_loss = 0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs}")
        for batch in pbar:
            t1 = batch['img_A'].to(device)
            t2 = batch['img_B'].to(device)
            l1 = batch['sem1'].to(device)
            l2 = batch['sem2'].to(device)
            gt_cd = batch['bcd'].to(device) 
            
            gt_bd = extract_boundary(gt_cd).to(device)
            
            optimizer.zero_grad()
            
            # [UPDATED] Use modern Autocast specifying 'cuda'
            with autocast('cuda', enabled=torch.cuda.is_available()):
                pred_ss1, pred_ss2, pred_cd, pred_bd = model(t1, t2)
                loss, loss_dict = criterion((pred_ss1, pred_ss2, pred_cd, pred_bd), (l1, l2, gt_cd, gt_bd))
            
            # Use Scaler for the backward pass and optimization
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            epoch_loss += loss.item()
            pbar.set_postfix({"Loss": f"{loss.item():.4f}", "CD_Loss": f"{loss_dict['cd']:.4f}"})
            
        scheduler.step()
        
        # Validation Step
        score = validate(model, val_loader, metrics, device, epoch)
        
        # Checkpoint Saving
        if score > best_score:
            best_score = score
            save_path = os.path.join(args.save_dir, f'boundnext_{args.model_type}_best.pth')
            torch.save(model.state_dict(), save_path)
            print(f"New Best Score: {best_score:.4f} (Saved to {save_path})")
        
        torch.save(model.state_dict(), os.path.join(args.save_dir, f'boundnext_{args.model_type}_last.pth'))

def validate(model, loader, metrics, device, epoch):
    model.eval()
    metrics.reset()
    
    with torch.no_grad():
        for batch in tqdm(loader, desc="Validating"):
            t1 = batch['img_A'].to(device)
            t2 = batch['img_B'].to(device)
            l1 = batch['sem1'].to(device)
            l2 = batch['sem2'].to(device)
            gt_cd = batch['bcd'].to(device)
            
            # [UPDATED] Modern Autocast for validation
            with autocast('cuda', enabled=torch.cuda.is_available()):
                pred_ss1, pred_ss2, pred_cd, _ = model(t1, t2)
            
            p_ss1 = torch.argmax(pred_ss1, dim=1)
            p_ss2 = torch.argmax(pred_ss2, dim=1)
            p_cd = (torch.sigmoid(pred_cd) > 0.5).long().squeeze(1)
            
            metrics.update(p_ss1, p_ss2, p_cd, l1, l2, gt_cd)
            
    results = metrics.compute()
    print(f"\n[Validation Epoch {epoch+1}]")
    print(f"SeK: {results['sek']:.4f} | F1_BCD: {results['f1_bcd']:.4f} | mIoU: {results['miou']:.4f} | Score: {results['score']:.4f}")
    
    return results['score']

if __name__ == '__main__':
    args = parse_args()
    train(args)
