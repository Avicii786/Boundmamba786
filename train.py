import os
import argparse
import random
import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import warnings
import logging

# 1. Total Silence for "Garbage" Logs
warnings.filterwarnings("ignore")
logging.getLogger("pytorch_lightning").setLevel(logging.ERROR)

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint

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
        
        # Initialize Architecture
        num_classes = 7 if args.dataset_name.upper() == 'SECOND' else 5
        self.model = BoundNeXt(
            num_classes=num_classes, 
            pretrained_path=args.weights, 
            model_type=args.model_type
        )
        
        self.criterion = BoundMambaLoss()
        self.metrics = SCDMetrics(num_classes=num_classes)

    def forward(self, t1, t2):
        return self.model(t1, t2)

    def on_train_start(self):
        if self.trainer.is_global_zero:
            print("\n" + "="*110)
            print(f"{'Epoch':^7} | {'T-Loss':^8} | {'S-Loss':^8} | {'B-Loss':^8} | {'V-Loss':^8} | {'SeK':^7} | {'F1-BCD':^7} | {'Score':^7}")
            print("-" * 110)

    def on_train_epoch_start(self):
        # Backbone Freezing Logic
        freeze_epochs = self.args.freeze_epochs
        if freeze_epochs > 0:
            should_freeze = self.current_epoch < freeze_epochs
            for param in self.model.encoder.stem.parameters(): param.requires_grad = not should_freeze
            for param in self.model.encoder.stages.parameters(): param.requires_grad = not should_freeze

    def training_step(self, batch, batch_idx):
        t1, t2, l1, l2, gt_cd = batch['img_A'], batch['img_B'], batch['sem1'], batch['sem2'], batch['bcd']
        gt_bd = extract_boundary(gt_cd)
        
        pred_ss1, pred_ss2, pred_cd, pred_bd = self(t1, t2)
        loss, loss_dict = self.criterion((pred_ss1, pred_ss2, pred_cd, pred_bd), (l1, l2, gt_cd, gt_bd))
        
        # Log for table retrieval at end of epoch
        self.log('t_loss', loss, on_step=False, on_epoch=True, sync_dist=True)
        self.log('s_loss', loss_dict['ss'], on_step=False, on_epoch=True, sync_dist=True)
        self.log('b_loss', loss_dict['cd'], on_step=False, on_epoch=True, sync_dist=True)
        
        # Heartbeat
        if batch_idx % 100 == 0 and self.trainer.is_global_zero:
            print(".", end="", flush=True)
            
        return loss

    def validation_step(self, batch, batch_idx):
        t1, t2, l1, l2, gt_cd = batch['img_A'], batch['img_B'], batch['sem1'], batch['sem2'], batch['bcd']
        
        # [FIXED BUG]: Properly extract gt_bd and capture pred_bd from the model
        gt_bd = extract_boundary(gt_cd)
        pred_ss1, pred_ss2, pred_cd, pred_bd = self(t1, t2)
        
        # Pass 4D pred_bd and 3D gt_bd into criterion correctly
        loss, _ = self.criterion((pred_ss1, pred_ss2, pred_cd, pred_bd), (l1, l2, gt_cd, gt_bd))
        
        self.log('val_loss', loss, on_epoch=True, sync_dist=True)
        
        p_ss1, p_ss2 = torch.argmax(pred_ss1, dim=1), torch.argmax(pred_ss2, dim=1)
        p_cd = (torch.sigmoid(pred_cd) > 0.5).long().squeeze(1)
        self.metrics.update(p_ss1, p_ss2, p_cd, l1, l2, gt_cd)

    def on_validation_epoch_end(self):
        # DDP Sync
        device = self.device
        h1 = self.trainer.strategy.reduce(torch.tensor(self.metrics.hist_sem1, device=device), "sum")
        h2 = self.trainer.strategy.reduce(torch.tensor(self.metrics.hist_sem2, device=device), "sum")
        hb = self.trainer.strategy.reduce(torch.tensor(self.metrics.hist_bcd, device=device), "sum")
        self.metrics.hist_sem1, self.metrics.hist_sem2, self.metrics.hist_bcd = h1.cpu().numpy(), h2.cpu().numpy(), hb.cpu().numpy()
        
        res = self.metrics.compute()
        self.log('val_score', res['score'], sync_dist=False, rank_zero_only=True)
        
        # Table Row Printing
        if self.trainer.is_global_zero:
            if self.trainer.state.stage == 'sanity_check':
                return
            
            def gv(k): 
                v = self.trainer.callback_metrics.get(k, 0.0)
                return v.item() if isinstance(v, torch.Tensor) else v

            print(f"\r{self.current_epoch:^7} | {gv('t_loss'):^8.4f} | {gv('s_loss'):^8.4f} | {gv('b_loss'):^8.4f} | {gv('val_loss'):^8.4f} | {res['sek']:^7.4f} | {res['f1_bcd']:^7.4f} | {res['score']:^7.4f}")
        
        self.metrics.reset()

    def configure_optimizers(self):
        # SOTA Adjustment: Lower LR, Higher Weight Decay
        optimizer = optim.AdamW(self.parameters(), lr=self.args.lr, weight_decay=0.05)
        # Cosine Annealing with Warm Restarts
        scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=20, T_mult=2)
        return [optimizer], [scheduler]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', type=str, required=True)
    parser.add_argument('--dataset_name', type=str, default='SECOND')
    parser.add_argument('--model_type', type=str, default='convnextv2_base')
    parser.add_argument('--weights', type=str, default=None)
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--epochs', type=int, default=150)
    parser.add_argument('--freeze_epochs', type=int, default=5)
    parser.add_argument('--lr', type=float, default=1e-4) 
    parser.add_argument('--save_dir', type=str, default='/kaggle/working/checkpoints')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    pl.seed_everything(args.seed, workers=True)
    os.makedirs(args.save_dir, exist_ok=True)

    train_loader = DataLoader(SCDDataset(root=args.data_root, mode='train', dataset_name=args.dataset_name, patch_mode=False), 
                              batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True, drop_last=True)
    val_loader = DataLoader(SCDDataset(root=args.data_root, mode='val', dataset_name=args.dataset_name, patch_mode=False), 
                            batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True)

    model = BoundNeXtLightning(args)
    checkpoint_callback = ModelCheckpoint(dirpath=args.save_dir, monitor="val_score", mode="max", save_last=True, filename="best_model")

    trainer = pl.Trainer(
        devices="auto", accelerator="gpu", strategy="ddp_find_unused_parameters_true",  
        max_epochs=args.epochs, precision="16-mixed", callbacks=[checkpoint_callback], 
        enable_progress_bar=False, logger=False,
        sync_batchnorm=True 
    )

    if trainer.is_global_zero:
        print(f"\n🚀 BoundNeXt Training: {args.model_type.upper()} | SyncBN: ON | ASPP: ON")
    trainer.fit(model, train_loader, val_loader)

if __name__ == '__main__':
    main()
