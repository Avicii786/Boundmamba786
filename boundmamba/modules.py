import torch
import torch.nn as nn
import torch.nn.functional as F

# ==============================================================================
# 1. SC_UP: Semantic Correlation & Unchanged-Prior Module
# ==============================================================================
class SC_UP_Module(nn.Module):
    def __init__(self, in_dim):
        super().__init__()
        # Calculate Similarity Map
        self.sim_conv = nn.Sequential(
            nn.Conv2d(in_dim * 2, in_dim // 2, 1),
            nn.BatchNorm2d(in_dim // 2),
            nn.ReLU(),
            nn.Conv2d(in_dim // 2, 1, 1),
            nn.Sigmoid()
        )
        self.proj = nn.Conv2d(in_dim, in_dim, 1)

    def forward(self, f1, f2):
        sim_map = self.sim_conv(torch.cat([f1, f2], dim=1))
        # Unchanged Prior: High similarity -> Low weight
        change_attn = 1.0 - sim_map
        
        f1_refined = f1 * change_attn + self.proj(f1)
        f2_refined = f2 * change_attn + self.proj(f2)
        
        return f1_refined, f2_refined

# ==============================================================================
# 2. BGI: Boundary-Gated Interaction Module (Your Core Innovation)
# ==============================================================================
class BGI_Module(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.diff_conv = nn.Sequential(
            nn.Conv2d(dim, dim, 3, padding=1, groups=dim), # Depthwise
            nn.BatchNorm2d(dim),
            nn.Conv2d(dim, dim, 1),
            nn.BatchNorm2d(dim),
            nn.ReLU()
        )
        self.gate_conv = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 1, 1),
            nn.Sigmoid()
        )

    def forward(self, f_ss1, f_ss2, f_cd, boundary_map):
        sem_diff = f_ss1 - f_ss2
        
        # Resize boundary map if needed
        if boundary_map.shape[2:] != f_cd.shape[2:]:
            b_map = F.interpolate(boundary_map, size=f_cd.shape[2:], mode='bilinear')
        else:
            b_map = boundary_map
            
        gate = self.gate_conv(b_map)
        
        # Inject semantic difference, amplified by boundary presence
        injection = self.diff_conv(sem_diff) * (1 + gate)
        
        return f_cd + injection

# ==============================================================================
# 3. UWFF: Uncertainty-Weighted Feature Fusion Head
# ==============================================================================
class UWFF_Head(nn.Module):
    def __init__(self, in_dim, num_classes_ss=7):
        super().__init__()
        self.ss1_head = nn.Sequential(
            nn.Conv2d(in_dim, in_dim // 2, 3, padding=1),
            nn.BatchNorm2d(in_dim // 2),
            nn.ReLU(),
            nn.Conv2d(in_dim // 2, num_classes_ss, 1)
        )
        self.ss2_head = nn.Sequential(
            nn.Conv2d(in_dim, in_dim // 2, 3, padding=1),
            nn.BatchNorm2d(in_dim // 2),
            nn.ReLU(),
            nn.Conv2d(in_dim // 2, num_classes_ss, 1)
        )
        self.cd_head = nn.Sequential(
            nn.Conv2d(in_dim, in_dim // 2, 3, padding=1),
            nn.BatchNorm2d(in_dim // 2),
            nn.ReLU(),
            nn.Conv2d(in_dim // 2, 1, 1)
        )
        
    def get_entropy(self, logits):
        p = torch.softmax(logits, dim=1)
        log_p = F.log_softmax(logits, dim=1)
        entropy = -torch.sum(p * log_p, dim=1, keepdim=True)
        return entropy / torch.log(torch.tensor(logits.shape[1], dtype=torch.float, device=logits.device))

    def forward(self, x_ss1, x_ss2, x_cd):
        logit_ss1 = self.ss1_head(x_ss1)
        logit_ss2 = self.ss2_head(x_ss2)
        logit_cd = self.cd_head(x_cd)
        
        # Detach entropy to avoid gradient issues
        h1 = self.get_entropy(logit_ss1).detach()
        h2 = self.get_entropy(logit_ss2).detach()
        avg_h = (h1 + h2) / 2.0
        
        # Fusion: Trust raw CD more when SS is uncertain
        final_cd = logit_cd * (1 + avg_h)
        
        return logit_ss1, logit_ss2, final_cd