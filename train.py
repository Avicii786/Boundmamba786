import os
import glob
import argparse
import torch
import numpy as np
import torch.optim as optim
from torch.utils.data import DataLoader
import warnings
import logging
import gc

warnings.filterwarnings("ignore")
logging.getLogger("pytorch_lightning").setLevel(logging.ERROR)

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, Callback

from boundmamba import BoundNeXt, BoundMambaLoss
from boundmamba.utils import extract_boundary
from boundmamba.metrics import SCDMetrics
from dataset import SCDDataset

# =====================================================================
# [NEW] Bulletproof Checkpoint Cleanup
# Forcibly deletes any extra .ckpt files that PyTorch Lightning leaves behind
# =====================================================================
class CleanupCallback(Callback):
    def on_train_epoch_end(self, trainer, pl_module):
        if trainer.is_global_zero:
            ckpt_dir = trainer.checkpoint_callback.dirpath
            best_path = trainer.checkpoint_callback.best_model_path
            
            # Find all checkpoints in the directory
            all_ckpts = glob.glob(os.path.join(ckpt_dir, "*.ckpt"))
            for ckpt in all_ckpts:
                if ckpt != best_path:
                    try:
                        os.remove(ckpt)
                    except Exception:
                        pass
            
            # Force RAM cleanup to protect Kaggle's /tmp Docker memory
            gc.collect()
            torch.cuda.empty_cache()

class BoundNeXtLightning(pl.LightningModule):
    def __init__(self, args):
        super().__init__()
        self.save_hyperparameters()
        self.args = args
        
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
            print(f"{'Epoch':^7} | {'T-Loss':^8} | {'S-Loss':^8} | {'B-Loss':^8} | {'V-Loss':^8} | {'SeK':^7} | {'F1-BCD':^7} | {'mIoU':^7} | {'Score':^7}")
            print("-" * 110)

    def on_train_epoch_start(self):
        freeze_epochs = self.args.freeze_epochs
        if freeze_epochs > 0:
            should_freeze = self.current_epoch < freeze_epochs
            
            for param in self.model.encoder.stem.parameters(): 
                param.requires_grad = not should_freeze
            for param in self.model.encoder.stages.parameters(): 
                param.requires_grad = not should_freeze
            
            if self.current_epoch == 0 and self.trainer.is_global_zero:
                print(f"❄️  [Warm-up] Freezing ConvNeXtV2 backbone for {freeze_epochs} epochs...\n")
            elif self.current_epoch == freeze_epochs and self.trainer.is_global_zero:
                print(f"\n🔥 [Fine-Tuning] Unfreezing backbone for end-to-end training!")
                print("-" * 110)
                print(f"{'Epoch':^7} | {'T-Loss':^8} | {'S-Loss':^8} | {'B-Loss':^8} | {'V-Loss':^8} | {'SeK':^7} | {'F1-BCD':^7} | {'mIoU':^7} | {'Score':^7}")
                print("-" * 110)

    def training_step(self, batch, batch_idx):
        t1, t2, l1, l2, gt_cd = batch['img_A'], batch['img_B'], batch['sem1'], batch['sem2'], batch['bcd']
        gt_bd = extract_boundary(gt_cd)
        
        pred_ss1, pred_ss2, pred_cd, pred_bd = self(t1, t2)
        loss, loss_dict = self.criterion((pred_ss1, pred_ss2, pred_cd, pred_bd), (l1, l2, gt_cd, gt_bd))
        
        self.log('t_loss', loss, on_step=False, on_epoch=True, sync_dist=True)
        self.log('s_loss', loss_dict['ss'], on_step=False, on_epoch=True, sync_dist=True)
        self.log('b_loss', loss_dict['cd'], on_step=False, on_epoch=True, sync_dist=True)
        return loss

    def validation_step(self, batch, batch_idx):
        t1, t2, l1, l2, gt_cd = batch['img_A'], batch['img_B'], batch['sem1'], batch['sem2'], batch['bcd']
        pred_ss1, pred_ss2, pred_cd, pred_bd = self(t1, t2)
        
        loss, _ = self.criterion((pred_ss1, pred_ss2, pred_cd, pred_bd), (l1, l2, gt_cd, extract_boundary(gt_cd)))
        self.log('val_loss', loss, on_epoch=True, sync_dist=True)
        
        p_ss1, p_ss2 = torch.argmax(pred_ss1, dim=1), torch.argmax(pred_ss2, dim=1)
        p_cd = (torch.sigmoid(pred_cd) > 0.5).long().squeeze(1)
        self.metrics.update(p_ss1, p_ss2, p_cd, l1, l2, gt_cd)

    def on_validation_epoch_end(self):
        device = self.device
        h1_t = torch.tensor(self.metrics.hist_sem1, device=device)
        h2_t = torch.tensor(self.metrics.hist_sem2, device=device)
        hb_t = torch.tensor(self.metrics.hist_bcd, device=device)

        h1_g = self.all_gather(h1_t)
        h2_g = self.all_gather(h2_t)
        hb_g = self.all_gather(hb_t)

        if h1_g.dim() > h1_t.dim():
            h1_t, h2_t, hb_t = h1_g.sum(dim=0), h2_g.sum(dim=0), hb_g.sum(dim=0)

        self.metrics.hist_sem1 = h1_t.cpu().numpy()
        self.metrics.hist_sem2 = h2_t.cpu().numpy()
        self.metrics.hist_bcd = hb_t.cpu().numpy()
        
        self.last_val_metrics = self.metrics.compute()
        self.log('val_score', self.last_val_metrics['score'], sync_dist=False, rank_zero_only=True)
        self.metrics.reset()

    def on_train_epoch_end(self):
        if self.trainer.is_global_zero and hasattr(self, 'last_val_metrics'):
            def gv(k): 
                v = self.trainer.callback_metrics.get(k, torch.tensor(0.0))
                return v.item() if isinstance(v, torch.Tensor) else v

            res = self.last_val_metrics
            print(f" {self.current_epoch:^7} | {gv('t_loss'):^8.4f} | {gv('s_loss'):^8.4f} | {gv('b_loss'):^8.4f} | {gv('val_loss'):^8.4f} | {res['sek']:^7.4f} | {res['f1_bcd']:^7.4f} | {res['miou']:^7.4f} | {res['score']:^7.4f}")

    def configure_optimizers(self):
        optimizer = optim.AdamW(self.parameters(), lr=self.args.lr, weight_decay=0.05)
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

    # [CRITICAL FIX] pin_memory=False stops the Docker /tmp daemon from crashing
    train_loader = DataLoader(SCDDataset(root=args.data_root, mode='train', dataset_name=args.dataset_name, patch_mode=False), 
                              batch_size=args.batch_size, shuffle=True, num_workers=2, pin_memory=False, drop_last=True)
    val_loader = DataLoader(SCDDataset(root=args.data_root, mode='val', dataset_name=args.dataset_name, patch_mode=False), 
                            batch_size=args.batch_size, shuffle=False, num_workers=2, pin_memory=False)

    model = BoundNeXtLightning(args)
    
    # [CRITICAL FIX] Static filename. It will ONLY ever output "best_model.ckpt" and continuously overwrite it.
    checkpoint_callback = ModelCheckpoint(
        dirpath=args.save_dir, 
        monitor="val_score", 
        mode="max", 
        save_top_k=1, 
        save_last=False,
        filename="best_model", 
        save_weights_only=False 
    )

    trainer = pl.Trainer(
        devices="auto", accelerator="gpu", strategy="ddp_find_unused_parameters_true",  
        max_epochs=args.epochs, precision="16-mixed", 
        callbacks=[checkpoint_callback, CleanupCallback()], # Added forceful cleanup
        enable_progress_bar=False, logger=False,
        sync_batchnorm=True 
    )

    if trainer.is_global_zero:
        print(f"\n🚀 BoundNeXt Training: {args.model_type.upper()} | SyncBN: ON | ASPP: ON")
    trainer.fit(model, train_loader, val_loader)

if __name__ == '__main__':
    main()
