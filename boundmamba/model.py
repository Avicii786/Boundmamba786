import torch
import torch.nn as nn
import torch.nn.functional as F
from .backbone import SiameseConvNeXtV2
from .modules import SC_UP_Module, BGI_Module, UWFF_Head

# =====================================================================
# [SOTA NOVELTY] Boundary-Conditioned Temporal Fusion (BCTF)
# With FP16 Numerical Stability Patches for Mixed Precision Training
# =====================================================================
class BoundaryConditionedFusion(nn.Module):
    def __init__(self, in_channels, num_heads=8):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = in_channels // num_heads
        self.scale = self.head_dim ** -0.5

        # [STABILITY FIX 1] Input Normalization to prevent variance explosion
        self.norm_in = nn.BatchNorm2d(in_channels * 2)

        # Projects concatenated temporal features into Q, K, V
        self.qkv_conv = nn.Conv2d(in_channels * 2, in_channels * 3, kernel_size=1, bias=False)
        
        # Projects the 1-channel boundary map into a spatial weighting matrix
        self.boundary_proj = nn.Sequential(
            nn.Conv2d(1, in_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(in_channels),
            nn.Sigmoid()
        )
        
        # Output projection
        self.proj = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.norm = nn.BatchNorm2d(in_channels)
        
        # Local residual path to preserve smooth gradient flow
        self.local_residual = nn.Sequential(
            nn.Conv2d(in_channels * 2, in_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(in_channels)
        )

    def forward(self, f1, f2, boundary_map):
        B, C, H, W = f1.shape
        
        # 1. Prepare the Boundary Prior
        b_down = F.interpolate(boundary_map, size=(H, W), mode='bilinear', align_corners=False)
        b_weight = self.boundary_proj(b_down) # Shape: [B, C, H, W]
        
        # 2. Extract Q, K, V from temporal features
        x = torch.cat([f1, f2], dim=1) # Shape: [B, 2C, H, W]
        
        # Apply Input Normalization
        x_norm = self.norm_in(x)
        
        qkv = self.qkv_conv(x_norm)    # Shape: [B, 3C, H, W]
        q, k, v = qkv.chunk(3, dim=1)  # Each is [B, C, H, W]
        
        # 3. Condition Queries and Keys with the Boundary Prior
        q = q * (1.0 + b_weight)
        k = k * (1.0 + b_weight)
        
        # 4. Reshape for Multi-Head Global Attention
        N = H * W
        q = q.view(B, self.num_heads, self.head_dim, N).transpose(-2, -1) # [B, heads, N, head_dim]
        k = k.view(B, self.num_heads, self.head_dim, N)                   # [B, heads, head_dim, N]
        v = v.view(B, self.num_heads, self.head_dim, N).transpose(-2, -1) # [B, heads, N, head_dim]
        
        # 5. Calculate Edge-Aware Attention Matrix
        # [STABILITY FIX 2] Apply scale to Q before matrix multiplication
        q = q * self.scale
        
        # [STABILITY FIX 3] Upcast to FP32 purely for the Softmax calculation to prevent NaN
        attn = (q.float() @ k.float()) # Output is temporarily float32
        attn = attn.softmax(dim=-1).to(v.dtype) # Softmax applied safely, cast back to mixed precision
        
        # 6. Apply Attention to Values
        out = (attn @ v).transpose(-2, -1).reshape(B, C, H, W) # [B, C, H, W]
        
        # 7. Final Projection and Residual Fusion
        out = self.norm(self.proj(out))
        local_feat = self.local_residual(x)
        
        return F.relu(out + local_feat, inplace=True)

class DecoderBlock(nn.Module):
    def __init__(self, in_ch, skip_ch, out_ch):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch + skip_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x, skip=None):
        x = self.up(x)
        if skip is not None:
            if x.shape[2:] != skip.shape[2:]:
                x = F.interpolate(x, size=skip.shape[2:], mode='bilinear')
            x = torch.cat([x, skip], dim=1)
        return self.conv(x)

class BoundNeXt(nn.Module):
    def __init__(self, num_classes=7, pretrained_path=None, model_type='convnextv2_base'):
        super().__init__()
        
        self.encoder = SiameseConvNeXtV2(
            model_type=model_type,
            checkpoint_path=pretrained_path,
            drop_path_rate=0.3 
        )
        dims = self.encoder.dims 
        
        self.sc_up = SC_UP_Module(dims[3])
        
        # Initialize the Novel Boundary-Conditioned Fusion Block
        self.temporal_fusion = BoundaryConditionedFusion(in_channels=dims[3], num_heads=8)
        
        self.boundary_head = nn.Sequential(
            nn.Conv2d(dims[0], 64, 3, padding=1),
            nn.BatchNorm2d(64), nn.ReLU(),
            nn.Conv2d(64, 1, 1)
        )

        self.dec_ss1_3 = DecoderBlock(dims[3], dims[2], dims[2])
        self.dec_ss1_2 = DecoderBlock(dims[2], dims[1], dims[1])
        self.dec_ss1_1 = DecoderBlock(dims[1], dims[0], dims[0])
        
        self.dec_ss2_3 = DecoderBlock(dims[3], dims[2], dims[2])
        self.dec_ss2_2 = DecoderBlock(dims[2], dims[1], dims[1])
        self.dec_ss2_1 = DecoderBlock(dims[1], dims[0], dims[0])
        
        self.dec_cd_3 = DecoderBlock(dims[3], 0, dims[2])
        self.dec_cd_2 = DecoderBlock(dims[2], 0, dims[1])
        self.dec_cd_1 = DecoderBlock(dims[1], 0, dims[0])
        
        self.bgi_3 = BGI_Module(dims[2])
        self.bgi_2 = BGI_Module(dims[1])
        self.bgi_1 = BGI_Module(dims[0])
        
        self.head = UWFF_Head(dims[0], num_classes)

    def forward(self, t1, t2):
        img_size = t1.shape[2:]
        
        f1_list, f2_list = self.encoder(t1, t2)
        
        # 1. Generate Boundary Prior
        diff_low = torch.abs(f1_list[0] - f2_list[0])
        boundary_logits = self.boundary_head(diff_low)
        boundary_map = torch.sigmoid(boundary_logits)
        
        f1_3, f2_3 = self.sc_up(f1_list[3], f2_list[3])
        
        # 2. Pass Temporal Features AND Boundary Prior into Attention Block
        cd_3 = self.temporal_fusion(f1_3, f2_3, boundary_map)
        
        x1 = self.dec_ss1_3(f1_3, f1_list[2])
        x2 = self.dec_ss2_3(f2_3, f2_list[2])
        cd_x = self.dec_cd_3(cd_3)
        cd_x = self.bgi_3(x1, x2, cd_x, boundary_map) 
        
        x1 = self.dec_ss1_2(x1, f1_list[1])
        x2 = self.dec_ss2_2(x2, f2_list[1])
        cd_x = self.dec_cd_2(cd_x)
        cd_x = self.bgi_2(x1, x2, cd_x, boundary_map) 
        
        x1 = self.dec_ss1_1(x1, f1_list[0])
        x2 = self.dec_ss2_1(x2, f2_list[0])
        cd_x = self.dec_cd_1(cd_x)
        cd_x = self.bgi_1(x1, x2, cd_x, boundary_map) 
        
        out_ss1, out_ss2, out_cd = self.head(x1, x2, cd_x)
        
        out_ss1 = F.interpolate(out_ss1, size=img_size, mode='bilinear')
        out_ss2 = F.interpolate(out_ss2, size=img_size, mode='bilinear')
        out_cd = F.interpolate(out_cd, size=img_size, mode='bilinear')
        out_bd = F.interpolate(boundary_logits, size=img_size, mode='bilinear')
        
        return out_ss1, out_ss2, out_cd, out_bd
