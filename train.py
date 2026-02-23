import os
import argparse
import random
import numpy as np
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

    def on_train_epoch_start(self):
        freeze_epochs = self.args.freeze_epochs
        if freeze_epochs > 0:
            if self.current_epoch < freeze_epochs:
                if self.current_epoch == 0 and self.trainer.is_global_zero:
                    print(f"\n❄️  [Warm-up] Freezing ConvNeXtV2 backbone for the first {freeze_epochs} epochs...")
                
                for param in self.model.encoder.stem.parameters():
                    param.requires_grad = False
                for param in self.model.encoder.stages.parameters():
                    param.requires_grad = False
                    
            elif self.current_epoch == freeze_epochs:
                if self.trainer.is_global_zero:
                    print(f"\n🔥 [Fine-Tuning] Unfreezing ConvNeXtV2 backbone for end-to-end training!")
                
                for param in self.model.encoder.stem.parameters():
                    param.requires_grad = True
                for param in self.model.encoder.stages.parameters():
                    param.requires_grad = True

    def training_step(self, batch, batch_idx):
        t1, t2 = batch['img_A'], batch['img_B']
        l1, l2, gt_cd = batch['sem1'], batch['sem2'], batch['bcd']
        
        gt_bd = extract_boundary(gt_cd)
        
        pred_ss1, pred_ss2, pred_cd, pred_bd = self(t1, t2)
        
        loss, loss_dict = self.criterion(
            (pred_ss1, pred_ss2, pred_cd, pred_bd), 
            (l1, l2, gt_cd, gt_bd)
        )
        
        # Logging with on_epoch=True appends "_epoch" to the keys for end-of-epoch retrieval
        self.log('train_loss', loss, on_step=True, on_epoch=True, sync_dist=True, batch_size=self.args.batch_size)
        self.log('sem_loss', loss_dict['ss'], on_step=True, on_epoch=True, sync_dist=True, batch_size=self.args.batch_size)
        self.log('bcd_loss', loss_dict['cd'], on_step=True, on_epoch=True, sync_dist=True, batch_size=self.args.batch_size)
        
        return loss

    # =====================================================================
    # [NEW] CLEAN TRAINING LOG OUTPUT
    # =====================================================================
    def on_train_epoch_end(self):
        if self.trainer.is_global_zero:
            # Helper safely extracts values whether they are tensors or floats
            def get_val(key):
                val = self.trainer.callback_metrics.get(key, 0.0)
                return val.item() if isinstance(val, torch.Tensor) else val
            
            t_loss = get_val('train_loss_epoch')
            s_loss = get_val('sem_loss_epoch')
            c_loss = get_val('bcd_loss_epoch')
            
            print(f"\n[Train Epoch {self.current_epoch}] Total Loss: {t_loss:.4f} | SEM Loss: {s_loss:.4f} | BCD Loss: {c_loss:.4f}")

    def validation_step(self, batch, batch_idx):
        t1, t2 = batch['img_A'], batch['img_B']
        l1, l2, gt_cd = batch['sem1'], batch['sem2'], batch['bcd']
        
        gt_bd = extract_boundary(gt_cd)
        pred_ss1, pred_ss2, pred_cd, pred_bd = self(t1, t2)
        
        loss, _ = self.criterion(
            (pred_ss1, pred_ss2, pred_cd, pred_bd), 
            (l1, l2, gt_cd, gt_bd)
        )
        
        self.log('val_loss', loss, on_epoch=True, sync_dist=True, batch_size=self.args.batch_size)
        
        p_ss1 = torch.argmax(pred_ss1, dim=1)
        p_ss2 = torch.argmax(pred_ss2, dim=1)
        p_cd = (torch.sigmoid(pred_cd) > 0.5).long().squeeze(1)
        
        self.metrics.update(p_ss1, p_ss2, p_cd, l1, l2, gt_cd)

    def on_validation_epoch_end(self):
        device = self.device
        hist_sem1_t = torch.tensor(self.metrics.hist_sem1, device=device)
        hist_sem2_t = torch.tensor(self.metrics.hist_sem2, device=device)
        hist_bcd_t = torch.tensor(self.metrics.hist_bcd, device=device)

        hist_sem1_t = self.trainer.strategy.reduce(hist_sem1_t, reduce_op="sum")
        hist_sem2_t = self.trainer.strategy.reduce(hist_sem2_t, reduce_op="sum")
        hist_bcd_t = self.trainer.strategy.reduce(hist_bcd_t, reduce_op="sum")

        self.metrics.hist_sem1 = hist_sem1_t.cpu().numpy()
        self.metrics.hist_sem2 = hist_sem2_t.cpu().numpy()
        self.metrics.hist_bcd = hist_bcd_t.cpu().numpy()
        
        res = self.metrics.compute()
        
        val_loss = self.trainer.callback_metrics.get('val_loss', torch.tensor(0.0))
        val_loss_val = val_loss.item() if isinstance(val_loss, torch.Tensor) else val_loss
        
        # Clean Output for Validation
        if self.trainer.is_global_zero:
            print(f"[Valid Epoch {self.current_epoch}] Val Loss: {val_loss_val:.4f} | SeK: {res['sek']:.4f} | F1_BCD: {res['f1_bcd']:.4f} | mIoU: {res['miou']:.4f} | Score: {res['score']:.4f}")
            print("-" * 75)
        
        self.log('val_score', res['score'], sync_dist=False, rank_zero_only=True)
        self.log('val_miou', res['miou'], sync_dist=False, rank_zero_only=True)
        
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
    parser.add_argument('--freeze_epochs', type=int, default=5, help='Number of epochs to freeze pretrained backbone')
    parser.add_argument('--lr', type=float, default=0.0003)
    parser.add_argument('--save_dir', type=str, default='/kaggle/working/checkpoints')
    parser.add_argument('--seed', type=int, default=42)
    return parser.parse_args()

def main():
    args = parse_args()
    
    pl.seed_everything(args.seed, workers=True)
    os.makedirs(args.save_dir, exist_ok=True)

    train_set = SCDDataset(root=args.data_root, mode='train', dataset_name=args.dataset_name, patch_mode=False)
    val_set = SCDDataset(root=args.data_root, mode='val', dataset_name=args.dataset_name, patch_mode=False)
    
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True)

    model = BoundNeXtLightning(args)

    checkpoint_callback = ModelCheckpoint(
        dirpath=args.save_dir,
        filename=f"boundnext_{args.model_type}" + "_{epoch:02d}_{val_score:.4f}",
        save_top_k=1,
        monitor="val_score",
        mode="max",
        save_last=True
    )
    lr_monitor = LearningRateMonitor(logging_interval='epoch')

    trainer = pl.Trainer(
        devices="auto",                  
        accelerator="gpu",
        strategy="ddp_find_unused_parameters_true",  
        max_epochs=args.epochs,
        precision="16-mixed",            
        callbacks=[checkpoint_callback, lr_monitor],
        enable_progress_bar=False,  # <--- [FIX] Disables the spammy TQDM bar for background runs!
        logger=False                # Optionally set to True if you want to use Tensorboard
    )

    if trainer.is_global_zero:
        print("\n🚀 Training Starting... (Progress bar disabled for clean Kaggle background logs)\n")

    trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)

if __name__ == '__main__':
    main()
