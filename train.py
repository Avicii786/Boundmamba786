import os
import argparse
import random
import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

# Imports from your package (Updated Imports)
from boundmamba.model import BoundMamba
from boundmamba.losses import JointLoss
from boundmamba.utils import extract_boundary
from boundmamba.metrics import SCDMetrics
from dataset import SCDDataset

def set_seed(seed=42):
    """
    Sets the seed for reproducibility across random, numpy, and torch.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # Ensure deterministic behavior for cuDNN
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print(f"Random Seed set to: {seed}")

def parse_args():
    parser = argparse.ArgumentParser(description="Train BoundMamba for Semantic Change Detection")
    parser.add_argument('--data_root', type=str, required=True, help='Path to SECOND or LandsatSCD dataset root')
    parser.add_argument('--dataset_name', type=str, default='SECOND', choices=['SECOND', 'LandsatSCD'])
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--lr', type=float, default=0.0003)
    parser.add_argument('--weights', type=str, default=None, help='Path to VMamba pretrained weights')
    parser.add_argument('--save_dir', type=str, default='./checkpoints')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility')
    return parser.parse_args()

def train(args):
    # 1. Set Seed & Device
    set_seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    os.makedirs(args.save_dir, exist_ok=True)

    # 2. Initialize Model & Loss
    # Get num_classes from dataset config
    num_classes = 7 if args.dataset_name == 'SECOND' else 5
    
    print(f"Initializing BoundMamba (Num Classes: {num_classes})...")
    model = BoundMamba(num_classes=num_classes, pretrained=args.weights).to(device)
    # Updated: Changed BoundMambaLoss() to JointLoss()
    criterion = JointLoss().to(device)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    
    metrics = SCDMetrics(num_classes=num_classes)

    # 3. DataLoaders (512x512 Configuration)
    # patch_mode=False ensures we load the full 512x512 images instead of 256x256 crops
    print(f"Loading {args.dataset_name} dataset with 512x512 resolution (patch_mode=False)...")
    
    train_set = SCDDataset(
        root=args.data_root, 
        mode='train', 
        dataset_name=args.dataset_name, 
        patch_mode=False  # Changed to False for 512x512 training
    )
    val_set = SCDDataset(
        root=args.data_root, 
        mode='val', 
        dataset_name=args.dataset_name, 
        patch_mode=False  # Changed to False for 512x512 validation
    )
    
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True)

    print(f"Train samples: {len(train_set)}, Val samples: {len(val_set)}")

    best_score = 0.0

    for epoch in range(args.epochs):
        model.train()
        epoch_loss = 0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs}")
        for batch in pbar:
            # Inputs
            t1 = batch['img_A'].to(device)
            t2 = batch['img_B'].to(device)
            
            # Ground Truth
            l1 = batch['sem1'].to(device)
            l2 = batch['sem2'].to(device)
            gt_cd = batch['bcd'].to(device) # Binary change mask
            
            # --- CRITICAL: Generate Boundary on the fly ---
            # The model needs boundaries for the BGI module supervision
            gt_bd = extract_boundary(gt_cd).to(device)
            
            optimizer.zero_grad()
            
            # Forward Pass
            # outputs: (ss1_logits, ss2_logits, cd_logits, bd_logits)
            pred_ss1, pred_ss2, pred_cd, pred_bd = model(t1, t2)
            
            # Calculate Loss
            loss, loss_dict = criterion(
                (pred_ss1, pred_ss2, pred_cd, pred_bd), 
                (l1, l2, gt_cd, gt_bd)
            )
            
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            pbar.set_postfix({"Loss": f"{loss.item():.4f}", "CD_Loss": f"{loss_dict['cd']:.4f}"})
            
        scheduler.step()
        
        # Validation Step
        score = validate(model, val_loader, metrics, device, epoch)
        
        # Checkpoint
        if score > best_score:
            best_score = score
            torch.save(model.state_dict(), os.path.join(args.save_dir, 'boundmamba_best.pth'))
            print(f"New Best Score: {best_score:.4f} (Saved)")
        
        torch.save(model.state_dict(), os.path.join(args.save_dir, 'boundmamba_last.pth'))

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
            
            pred_ss1, pred_ss2, pred_cd, _ = model(t1, t2)
            
            # Process Predictions for Metrics
            # Semantics: Argmax
            p_ss1 = torch.argmax(pred_ss1, dim=1)
            p_ss2 = torch.argmax(pred_ss2, dim=1)
            # Change: Threshold > 0
            p_cd = (torch.sigmoid(pred_cd) > 0.5).long().squeeze(1)
            
            metrics.update(p_ss1, p_ss2, p_cd, l1, l2, gt_cd)
            
    results = metrics.compute()
    print(f"\n[Validation Epoch {epoch+1}]")
    print(f"SeK: {results['sek']:.4f} | F1_BCD: {results['f1_bcd']:.4f} | mIoU: {results['miou']:.4f} | Score: {results['score']:.4f}")
    
    return results['score']

if __name__ == '__main__':
    args = parse_args()
    train(args)
