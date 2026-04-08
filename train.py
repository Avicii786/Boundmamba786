import os
import glob
import argparse
import torch
import numpy as np
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
import warnings
import logging
import gc

warnings.filterwarnings("ignore")
logging.getLogger("pytorch_lightning").setLevel(logging.ERROR)

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, Callback, EarlyStopping

from boundmamba import BoundNeXt, BoundMambaLoss
from boundmamba.utils import extract_composite_boundary
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
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

def bitemporal_mixup(t1, t2, l1, l2, bcd, alpha=1.0):
    if np.random.rand() > 0.25: 
        return t1, t2, l1, l2, bcd

    B, C, H, W = t1.shape
    index = torch.randperm(B).to(t1.device)

    lam = np.random.beta(alpha, alpha)
    cut_rat = np.clip(np.sqrt(1. - lam), 0.1, 0.6) 
    cut_w = int(W * cut_rat)
    cut_h = int(H * cut_rat)

    cx = np.random.randint(W)
    cy = np.random.randint(H)

    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)

    t1[:, :, bby1:bby2, bbx1:bbx2] = t1[index, :, bby1:bby2, bbx1:bbx2]
    t2[:, :, bby1:bby2, bbx1:bbx2] = t2[index, :, bby1:bby2, bbx1:bbx2]
    
    l1[:, bby1:bby2, bbx1:bbx2] = l1[index, bby1:bby2, bbx1:bbx2]
    l2[:, bby1:bby2, bbx1:bbx2] = l2[index, bby1:bby2, bbx1:bbx2]
    bcd[:, bby1:bby2, bbx1:bbx2] = bcd[index, bby1:bby2, bbx1:bbx2]

    return t1, t2, l1, l2, bcd

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
        
        t1, t2, l1, l2, gt_cd = bitemporal_mixup(t1, t2, l1, l2, gt_cd)
        gt_bd = extract_composite_boundary(gt_cd, l1, l2)
        
        outputs, aux_outputs = self(t1, t2)
        pred_ss1, pred_ss2, pred_cd, pred_bd = outputs
        
        loss_main, loss_dict = self.criterion(outputs, (l1, l2, gt_cd, gt_bd))
        
        aux_shape = aux_outputs[0].shape[2:]
        aux_l1 = F.interpolate(l1.unsqueeze(1).float(), size=aux_shape, mode='nearest').squeeze(1).long()
        aux_l2 = F.interpolate(l2.unsqueeze(1).float(), size=aux_shape, mode='nearest').squeeze(1).long()
        aux_cd = F.interpolate(gt_cd.unsqueeze(1).float(), size=aux_shape, mode='nearest').squeeze(1).float()
        aux_bd = F.interpolate(gt_bd.unsqueeze(1).float(), size=aux_shape, mode='nearest').squeeze(1).float()
        
        loss_aux, _ = self.criterion((aux_outputs[0], aux_outputs[1], aux_outputs[2], aux_outputs[2]), (aux_l1, aux_l2, aux_cd, aux_bd))
        
        loss = loss_main + (0.4 * loss_aux)
        
        self.log('t_loss', loss, on_step=False, on_epoch=True, sync_dist=True)
        self.log('s_loss', loss_dict['ss'], on_step=False, on_epoch=True, sync_dist=True)
        self.log('b_loss', loss_dict['cd'], on_step=False, on_epoch=True, sync_dist=True)
        return loss

    def validation_step(self, batch, batch_idx):
        t1, t2, l1, l2, gt_cd = batch['img_A'], batch['img_B'], batch['sem1'], batch['sem2'], batch['bcd']
        
        gt_bd = extract_composite_boundary(gt_cd, l1, l2)
        
        logits_ss1, logits_ss2, logits_cd, logits_bd = self(t1, t2)
        loss, _ = self.criterion((logits_ss1, logits_ss2, logits_cd, logits_bd), (l1, l2, gt_cd, gt_bd))
        self.log('val_loss', loss, on_epoch=True, sync_dist=True)
        
        p_ss1 = torch.argmax(logits_ss1, dim=1)
        p_ss2 = torch.argmax(logits_ss2, dim=1)
        p_cd = (torch.sigmoid(logits_cd) > 0.5).long().squeeze(1)
        
        # --- [SOTA POST-PROCESSING FIX] Dual-Way Conflict Resolution ---
        
        # Rule 1: If Semantic heads predict the exact same class, it implies NO CHANGE.
        # The deep semantic network is highly accurate at rejecting pseudo-changes.
        # We override the binary change map to 0 in these regions.
        semantic_agree = (p_ss1 == p_ss2)
        p_cd = torch.where(semantic_agree, torch.zeros_like(p_cd), p_cd)

        # Rule 2: If the Change head still confidently says NO CHANGE (p_cd == 0),
        # but the semantic heads disagree, we force the semantic heads to agree 
        # using their highest combined ensemble confidence.
        logits_shared = logits_ss1 + logits_ss2
        p_shared = torch.argmax(logits_shared, dim=1)
        
        p_ss1 = torch.where(p_cd == 0, p_shared, p_ss1)
        p_ss2 = torch.where(p_cd == 0, p_shared, p_ss2)
        
        # ----------------------------------------------------------------
        
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
        
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='max', factor=0.5, patience=8, min_lr=1e-6, verbose=False 
        )
        
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val_score",
                "frequency": 1
            }
        }

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', type=str, required=True)
    parser.add_argument('--dataset_name', type=str, default='SECOND')
    parser.add_argument('--model_type', type=str, default='convnextv2_base')
    parser.add_argument('--weights', type=str, default=None)
    parser.add_argument('--batch_size', type=int, default=2) 
    parser.add_argument('--accumulate_grad_batches', type=int, default=4) 
    parser.add_argument('--epochs', type=int, default=100) 
    parser.add_argument('--freeze_epochs', type=int, default=5)
    parser.add_argument('--lr', type=float, default=1e-4) 
    parser.add_argument('--patience', type=int, default=15, help='Patience for early stopping')
    parser.add_argument('--save_dir', type=str, default='/kaggle/working/checkpoints')
    parser.add_argument('--seed', type=int, default=42)
    
    parser.add_argument('--accelerator', type=str, default='gpu', choices=['gpu', 'tpu', 'cpu', 'auto'])
    parser.add_argument('--devices', type=str, default='auto')
    args = parser.parse_args()

    pl.seed_everything(args.seed, workers=True)
    os.makedirs(args.save_dir, exist_ok=True)

    train_loader = DataLoader(SCDDataset(root=args.data_root, mode='train', dataset_name=args.dataset_name, patch_mode=False), 
                              batch_size=args.batch_size, shuffle=True, num_workers=2, pin_memory=False, drop_last=True)
    val_loader = DataLoader(SCDDataset(root=args.data_root, mode='val', dataset_name=args.dataset_name, patch_mode=False), 
                            batch_size=args.batch_size, shuffle=False, num_workers=2, pin_memory=False)

    model = BoundNeXtLightning(args)
    
    checkpoint_callback = ModelCheckpoint(
        dirpath=args.save_dir, 
        monitor="val_score", 
        mode="max", 
        save_top_k=1, 
        save_last=False, 
        filename="best_model", 
        save_weights_only=True
    )
    
    early_stop_callback = EarlyStopping(
        monitor="val_score",
        min_delta=0.0001,
        patience=args.patience,
        verbose=False, 
        mode="max"
    )

    strategy = "ddp_find_unused_parameters_true" if args.accelerator == 'gpu' else "auto"
    devices = int(args.devices) if args.devices.isdigit() else args.devices
    precision = "16-mixed" if args.accelerator == 'gpu' else "bf16-mixed"
    
    # [SOTA BATCH FIX] sync_batchnorm is no longer needed because we ripped out BatchNorm
    # entirely in favor of GroupNorm, which is intrinsically immune to multi-GPU synchronization issues.
    
    trainer = pl.Trainer(
        accelerator=args.accelerator, 
        devices=devices, 
        strategy=strategy,
        min_epochs = 65,
        max_epochs=args.epochs, 
        precision=precision,
        accumulate_grad_batches=args.accumulate_grad_batches,
        gradient_clip_val=1.0, 
        callbacks=[checkpoint_callback, early_stop_callback, CleanupCallback()], 
        enable_progress_bar=False, 
        logger=False
    )

    if trainer.is_global_zero:
        eff_bs = args.batch_size * args.accumulate_grad_batches * (2 if args.accelerator == 'gpu' else 8)
        print(f"\n🚀 BoundNeXt: {args.model_type.upper()} | Accel: {args.accelerator.upper()} | Eff. Batch: {eff_bs} | GroupNorm: ON")
        print(f"🛑 Early Stopping: ON (Patience: {args.patience} Epochs)")
    
    trainer.fit(model, train_loader, val_loader)

if __name__ == '__main__':
    main()
