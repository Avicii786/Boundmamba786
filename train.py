import os
import argparse
import torch
import torch.optim as optim
from torch.utils.data import DataLoader

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor

# Imports from your package
from boundmamba import BoundNeXt, BoundMambaLoss
from boundmamba.utils import extract_boundary
from boundmamba.metrics import SCDMetrics
from dataset import SCDDataset

class BoundNeXtLightning(pl.LightningModule):
    def __init__(self, args):
        super().__init__()
        self.save_hyperparameters()
        self.args = args
        
        # 1. Initialize Architecture
        num_classes = 7 if args.dataset_name.upper() == 'SECOND' else 5
        self.model = BoundNeXt(
            num_classes=num_classes, 
            pretrained_path=args.weights, 
            model_type=args.model_type
        )
        
        # 2. Loss & Metrics
        self.criterion = BoundMambaLoss()
        self.metrics = SCDMetrics(num_classes=num_classes)

    def forward(self, t1, t2):
        return self.model(t1, t2)

    def training_step(self, batch, batch_idx):
        t1, t2 = batch['img_A'], batch['img_B']
        l1, l2, gt_cd = batch['sem1'], batch['sem2'], batch['bcd']
        
        # On-the-fly boundary generation
        gt_bd = extract_boundary(gt_cd)
        
        # Forward pass
        pred_ss1, pred_ss2, pred_cd, pred_bd = self(t1, t2)
        
        # Loss
        loss, loss_dict = self.criterion(
            (pred_ss1, pred_ss2, pred_cd, pred_bd), 
            (l1, l2, gt_cd, gt_bd)
        )
        
        # Lightning handles logging automatically
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log('cd_loss', loss_dict['cd'], on_step=True, on_epoch=False, prog_bar=True, sync_dist=True)
        
        return loss

    def validation_step(self, batch, batch_idx):
        t1, t2 = batch['img_A'], batch['img_B']
        l1, l2, gt_cd = batch['sem1'], batch['sem2'], batch['bcd']
        
        pred_ss1, pred_ss2, pred_cd, _ = self(t1, t2)
        
        # Predictions
        p_ss1 = torch.argmax(pred_ss1, dim=1)
        p_ss2 = torch.argmax(pred_ss2, dim=1)
        p_cd = (torch.sigmoid(pred_cd) > 0.5).long().squeeze(1)
        
        # Update metrics locally on this GPU
        self.metrics.update(p_ss1, p_ss2, p_cd, l1, l2, gt_cd)

    def on_validation_epoch_end(self):
        # --- DDP Metric Aggregation Strategy ---
        # Because SCDMetrics uses numpy internally, we temporarily convert 
        # the confusion matrices to tensors, sum them across all GPUs, 
        # and then compute the final score on the main GPU (Rank 0).
        device = self.device
        hist_sem1_t = torch.tensor(self.metrics.hist_sem1, device=device)
        hist_sem2_t = torch.tensor(self.metrics.hist_sem2, device=device)
        hist_bcd_t = torch.tensor(self.metrics.hist_bcd, device=device)

        # Sum across all GPUs (using Lightning's strategy reducer)
        hist_sem1_t = self.trainer.strategy.reduce(hist_sem1_t, reduce_op="sum")
        hist_sem2_t = self.trainer.strategy.reduce(hist_sem2_t, reduce_op="sum")
        hist_bcd_t = self.trainer.strategy.reduce(hist_bcd_t, reduce_op="sum")

        if self.trainer.is_global_zero:
            self.metrics.hist_sem1 = hist_sem1_t.cpu().numpy()
            self.metrics.hist_sem2 = hist_sem2_t.cpu().numpy()
            self.metrics.hist_bcd = hist_bcd_t.cpu().numpy()
            
            res = self.metrics.compute()
            print(f"\n[Validation Epoch {self.current_epoch}]")
            print(f"SeK: {res['sek']:.4f} | F1_BCD: {res['f1_bcd']:.4f} | mIoU: {res['miou']:.4f} | Score: {res['score']:.4f}")
            
            # Log metrics so the Checkpoint Callback can track 'val_score'
            self.log('val_score', res['score'], rank_zero_only=True)
            self.log('val_miou', res['miou'], rank_zero_only=True)
        
        # Reset local metrics for the next epoch
        self.metrics.reset()

    def configure_optimizers(self):
        optimizer = optim.AdamW(self.parameters(), lr=self.args.lr, weight_decay=1e-4)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.args.epochs)
        return [optimizer], [scheduler]

def parse_args():
    parser = argparse.ArgumentParser(description="Train BoundNeXt with PyTorch Lightning")
    parser.add_argument('--data_root', type=str, required=True)
    parser.add_argument('--dataset_name', type=str, default='SECOND')
    parser.add_argument('--model_type', type=str, default='convnextv2_tiny')
    parser.add_argument('--weights', type=str, default=None)
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--lr', type=float, default=0.0003)
    parser.add_argument('--save_dir', type=str, default='/kaggle/working/checkpoints')
    parser.add_argument('--seed', type=int, default=42)
    return parser.parse_args()

def main():
    args = parse_args()
    
    # Lightning sets the seed securely across all workers and GPUs
    pl.seed_everything(args.seed, workers=True)
    os.makedirs(args.save_dir, exist_ok=True)

    # Setup DataLoaders
    train_set = SCDDataset(root=args.data_root, mode='train', dataset_name=args.dataset_name, patch_mode=False)
    val_set = SCDDataset(root=args.data_root, mode='val', dataset_name=args.dataset_name, patch_mode=False)
    
    # num_workers=4 is safe now because Lightning handles DDP processes efficiently
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True)

    # Initialize Model Wrapper
    model = BoundNeXtLightning(args)

    # Setup Callbacks
    checkpoint_callback = ModelCheckpoint(
        dirpath=args.save_dir,
        filename=f"boundnext_{args.model_type}" + "_{epoch:02d}_{val_score:.4f}",
        save_top_k=1,
        monitor="val_score",
        mode="max",
        save_last=True
    )
    lr_monitor = LearningRateMonitor(logging_interval='epoch')

    # Initialize PyTorch Lightning Trainer
    trainer = pl.Trainer(
        devices="auto",                  # Automatically detects 2 GPUs
        accelerator="gpu",
        strategy="ddp",                  # Distributed Data Parallel for high speed
        max_epochs=args.epochs,
        precision="16-mixed",            # Automatic Mixed Precision (AMP) enabled natively
        callbacks=[checkpoint_callback, lr_monitor],
        log_every_n_steps=10,
        enable_progress_bar=True
    )

    # Start Training
    trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)

if __name__ == '__main__':
    main()
