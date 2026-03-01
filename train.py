import os
import glob
import argparse
import torch
import numpy as np
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.transforms.functional as TF
import warnings
import logging
import gc

warnings.filterwarnings("ignore")
logging.getLogger("pytorch_lightning").setLevel(logging.ERROR)

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, Callback, StochasticWeightAveraging

from boundmamba import BoundNeXt, BoundMambaLoss
from boundmamba.utils import extract_boundary
from boundmamba.metrics import SCDMetrics
from dataset import SCDDataset

class CleanupCallback(Callback):
    def on_train_epoch_end(self, trainer, pl_module):
        if trainer.is_global_zero:
            ckpt_dir = trainer.checkpoint_callback.dirpath
            best_path = trainer.checkpoint_callback.best_model_path
            all_ckpts = glob.glob(os.path.join(ckpt_dir, "*.ckpt"))
            for ckpt in all_ckpts:
                if ckpt != best_path:
                    try: os.remove(ckpt)
                    except: pass
            gc.collect()
            torch.cuda.empty_cache()

class BoundNeXtLightning(pl.LightningModule):
    def __init__(self, args):
        super().__init__()
        self.save_hyperparameters()
        self.args = args
        num_classes = 7 if args.dataset_name.upper() == 'SECOND' else 5
        self.model = BoundNeXt(num_classes=num_classes, pretrained_path=args.weights, model_type=args.model_type)
        self.criterion = BoundMambaLoss(num_classes=num_classes)
        self.metrics = SCDMetrics(num_classes=num_classes)

    def forward(self, t1, t2): return self.model(t1, t2)

    def on_train_start(self):
        if self.trainer.is_global_zero:
            print("\n" + "="*110)
            print(f"{'Epoch':^7} | {'T-Loss':^8} | {'S-Loss':^8} | {'B-Loss':^8} | {'V-Loss':^8} | {'SeK':^7} | {'F1-BCD':^7} | {'mIoU':^7} | {'Score':^7}")
            print("-" * 110)

    def on_train_epoch_start(self):
        freeze_epochs = self.args.freeze_epochs
        if freeze_epochs > 0:
            should_freeze = self.current_epoch < freeze_epochs
            for param in self.model.encoder.stem.parameters(): param.requires_grad = not should_freeze
            for param in self.model.encoder.stages.parameters(): param.requires_grad = not should_freeze
            
            if self.current_epoch == 0 and self.trainer.is_global_zero:
                print(f"❄️  [Warm-up] Freezing ConvNeXtV2 backbone for {freeze_epochs} epochs...\n")
            elif self.current_epoch == freeze_epochs and self.trainer.is_global_zero:
                print(f"\n🔥 [Fine-Tuning] Unfreezing backbone for Discriminative LR training!")
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
        
        # =====================================================================
        # [SOTA FIX] TEST-TIME AUGMENTATION (TTA)
        # Calculates predictions normally, horizontally, and vertically, then averages.
        # =====================================================================
        # 1. Normal Pass
        logits_ss1, logits_ss2, logits_cd, logits_bd = self(t1, t2)
        
        # 2. Horizontal Flip Pass
        h_ss1, h_ss2, h_cd, _ = self(TF.hflip(t1), TF.hflip(t2))
        logits_ss1 += TF.hflip(h_ss1)
        logits_ss2 += TF.hflip(h_ss2)
        logits_cd += TF.hflip(h_cd)
        
        # 3. Vertical Flip Pass
        v_ss1, v_ss2, v_cd, _ = self(TF.vflip(t1), TF.vflip(t2))
        logits_ss1 += TF.vflip(v_ss1)
        logits_ss2 += TF.vflip(v_ss2)
        logits_cd += TF.vflip(v_cd)
        
        # Average the logits
        logits_ss1 /= 3.0
        logits_ss2 /= 3.0
        logits_cd /= 3.0

        # Calculate Validation Loss
        loss, _ = self.criterion((logits_ss1, logits_ss2, logits_cd, logits_bd), (l1, l2, gt_cd, extract_boundary(gt_cd)))
        self.log('val_loss', loss, on_epoch=True, sync_dist=True)
        
        # Discrete Predictions
        p_ss1 = torch.argmax(logits_ss1, dim=1)
        p_ss2 = torch.argmax(logits_ss2, dim=1)
        p_cd = (torch.sigmoid(logits_cd) > 0.5).long().squeeze(1)
        
        # SOTA Inference Task Alignment Trick
        p_ss2 = torch.where(p_cd == 0, p_ss1, p_ss2)
        
        self.metrics.update(p_ss1, p_ss2, p_cd, l1, l2, gt_cd)

    def on_validation_epoch_end(self):
        device = self.device
        h1_t = torch.tensor(self.metrics.hist_sem1, device=device)
        h2_t = torch.tensor(self.metrics.hist_sem2, device=device)
        hb_t = torch.tensor(self.metrics.hist_bcd, device=device)

        h1_g, h2_g, hb_g = self.all_gather(h1_t), self.all_gather(h2_t), self.all_gather(hb_t)
        if h1_g.dim() > h1_t.dim():
            h1_t, h2_t, hb_t = h1_g.sum(dim=0), h2_g.sum(dim=0), hb_g.sum(dim=0)

        self.metrics.hist_sem1, self.metrics.hist_sem2, self.metrics.hist_bcd = h1_t.cpu().numpy(), h2_t.cpu().numpy(), hb_t.cpu().numpy()
        self.last_val_metrics = self.metrics.compute()
        self.log('val_score', self.last_val_metrics['score'], sync_dist=False, rank_zero_only=True)
        self.metrics.reset()

    def on_train_epoch_end(self):
        if self.trainer.is_global_zero and hasattr(self, 'last_val_metrics'):
            def gv(k): return self.trainer.callback_metrics.get(k, torch.tensor(0.0)).item()
            res = self.last_val_metrics
            print(f" {self.current_epoch:^7} | {gv('t_loss'):^8.4f} | {gv('s_loss'):^8.4f} | {gv('b_loss'):^8.4f} | {gv('val_loss'):^8.4f} | {res['sek']:^7.4f} | {res['f1_bcd']:^7.4f} | {res['miou']:^7.4f} | {res['score']:^7.4f}")

    def configure_optimizers(self):
        backbone_params, decoder_params = [], []
        for name, param in self.model.named_parameters():
            if 'encoder' in name:
                backbone_params.append(param)
            else:
                decoder_params.append(param)
                
        optimizer = optim.AdamW([
            {'params': backbone_params, 'lr': self.args.lr * 0.1}, 
            {'params': decoder_params, 'lr': self.args.lr}
        ], weight_decay=0.05)
        
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.args.epochs, eta_min=1e-6)
        return [optimizer], [scheduler]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', type=str, required=True)
    parser.add_argument('--dataset_name', type=str, default='SECOND')
    parser.add_argument('--model_type', type=str, default='convnextv2_base')
    parser.add_argument('--weights', type=str, default=None)
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--epochs', type=int, default=100) 
    parser.add_argument('--freeze_epochs', type=int, default=5)
    parser.add_argument('--lr', type=float, default=1e-4) 
    parser.add_argument('--save_dir', type=str, default='/kaggle/working/checkpoints')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    pl.seed_everything(args.seed, workers=True)
    os.makedirs(args.save_dir, exist_ok=True)

    train_loader = DataLoader(SCDDataset(root=args.data_root, mode='train', dataset_name=args.dataset_name, patch_mode=False), 
                              batch_size=args.batch_size, shuffle=True, num_workers=2, pin_memory=False, drop_last=True)
    val_loader = DataLoader(SCDDataset(root=args.data_root, mode='val', dataset_name=args.dataset_name, patch_mode=False), 
                            batch_size=args.batch_size, shuffle=False, num_workers=2, pin_memory=False)

    model = BoundNeXtLightning(args)
    checkpoint_callback = ModelCheckpoint(dirpath=args.save_dir, monitor="val_score", mode="max", save_top_k=1, save_last=False, filename="best_model", save_weights_only=True)
    swa_callback = StochasticWeightAveraging(swa_lrs=1e-5, swa_epoch_start=0.6)

    trainer = pl.Trainer(
        devices="auto", accelerator="gpu", strategy="ddp_find_unused_parameters_true", 
        max_epochs=args.epochs, precision="16-mixed", 
        callbacks=[checkpoint_callback, CleanupCallback(), swa_callback], 
        enable_progress_bar=False, logger=False, sync_batchnorm=True
    )

    if trainer.is_global_zero:
        print(f"\n🚀 BoundNeXt: {args.model_type.upper()} | SyncBN: ON | TTA: ON | BCD-Focal: ON")
    trainer.fit(model, train_loader, val_loader)

if __name__ == '__main__':
    main()
