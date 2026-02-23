import numpy as np
import math

class SCDMetrics:
    """
    Computes Semantic Change Detection metrics exactly matching the formulas
    defined in the SCanNet / Bi-SRNet papers for the SECOND dataset.
    """
    def __init__(self, num_classes=7):
        self.num_classes = num_classes
        self.reset()

    def reset(self):
        # Semantic confusion matrices (T1 and T2)
        self.hist_sem1 = np.zeros((self.num_classes, self.num_classes))
        self.hist_sem2 = np.zeros((self.num_classes, self.num_classes))
        # Binary Change confusion matrix
        self.hist_bcd = np.zeros((2, 2))
        self.count = 0

    def _fast_hist(self, label, pred, n_class):
        mask = (label >= 0) & (label < n_class)
        hist = np.bincount(
            n_class * label[mask].astype(int) + pred[mask],
            minlength=n_class ** 2
        ).reshape(n_class, n_class)
        return hist

    def update(self, pred_sem1, pred_sem2, pred_bcd, label_sem1, label_sem2, label_bcd):
        def to_numpy(x): return x.detach().cpu().numpy().flatten()
        
        p_sem1 = to_numpy(pred_sem1)
        p_sem2 = to_numpy(pred_sem2)
        p_bcd = to_numpy(pred_bcd)
        
        l_sem1 = to_numpy(label_sem1)
        l_sem2 = to_numpy(label_sem2)
        l_bcd = to_numpy(label_bcd)

        self.hist_sem1 += self._fast_hist(l_sem1, p_sem1, self.num_classes)
        self.hist_sem2 += self._fast_hist(l_sem2, p_sem2, self.num_classes)
        self.hist_bcd += self._fast_hist(l_bcd, p_bcd, 2)
        self.count += 1

    def compute(self):
        # Aggregate semantic performance
        hist_sem = self.hist_sem1 + self.hist_sem2
        
        # ---------------------------------------------------------
        # 1. mIoU calculation exactly as defined in SCanNet (Eq 12, 13, 14)
        # ---------------------------------------------------------
        q00 = hist_sem[0, 0]
        
        # Eq 13: IoU_nc (Unchanged IoU)
        sum_i0 = hist_sem[:, 0].sum()
        sum_0j = hist_sem[0, :].sum()
        iou_nc = q00 / (sum_i0 + sum_0j - q00 + 1e-10)
        
        # Eq 14: IoU_c (Changed IoU)
        # Numerator: sum of true positives for ALL change classes combined
        sum_ij_c = hist_sem[1:, 1:].sum() 
        # Denominator: All pixels MINUS the true negative background pixels
        den_c = hist_sem.sum() - q00 
        iou_c = sum_ij_c / (den_c + 1e-10)
        
        # Eq 12: mIoU
        paper_miou = (iou_nc + iou_c) / 2.0
        
        # ---------------------------------------------------------
        # 2. SeK calculation exactly as defined in SCanNet (Eq 15, 16, 17)
        # ---------------------------------------------------------
        hist_n0 = hist_sem.copy()
        hist_n0[0, 0] = 0 # Zero out true negative unchanged
        
        sum_hat_q = hist_n0.sum()
        if sum_hat_q == 0:
            kappa_n0 = 0.0
        else:
            rho = np.diag(hist_n0).sum() / sum_hat_q
            eta = (hist_n0.sum(axis=1) * hist_n0.sum(axis=0)).sum() / (sum_hat_q ** 2)
            if eta == 1:
                kappa_n0 = 0.0
            else:
                kappa_n0 = (rho - eta) / (1 - eta)
                
        kappa_n0 = max(0, kappa_n0)
        sek = (kappa_n0 * math.exp(iou_c - 1)) # Note: iou_c is used here as per Eq 17

        # ---------------------------------------------------------
        # 3. F_scd calculation exactly as defined in SCanNet (Eq 18, 19, 20)
        # ---------------------------------------------------------
        correct_change = np.diag(hist_sem)[1:].sum()
        pred_change = hist_sem[:, 1:].sum()
        gt_change = hist_sem[1:, :].sum()
        
        p_scd = correct_change / (pred_change + 1e-10)
        r_scd = correct_change / (gt_change + 1e-10)
        
        if (p_scd + r_scd) == 0:
            f_scd = 0.0
        else:
            f_scd = 2 * p_scd * r_scd / (p_scd + r_scd)

        # ---------------------------------------------------------
        # 4. Binary Metrics (from the pure BCD head)
        # ---------------------------------------------------------
        tn, fp, fn, tp = self.hist_bcd.flatten()
        pre_bcd = tp / (tp + fp + 1e-10)
        rec_bcd = tp / (tp + fn + 1e-10)
        f1_bcd = 2 * (pre_bcd * rec_bcd) / (pre_bcd + rec_bcd + 1e-10)
        
        # Calculate standard OA just for logging
        oa = np.diag(hist_sem).sum() / (hist_sem.sum() + 1e-10)

        # Composite score
        score = 0.3 * paper_miou + 0.7 * sek

        return {
            "sek": sek,
            "f1_bcd": f1_bcd,
            "miou": paper_miou,  # Now correctly returns SOTA Binary-averaged mIoU
            "oa": oa,
            "f_scd": f_scd,
            "score": score
        }