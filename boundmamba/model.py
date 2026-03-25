import torch
import torch.nn as nn
import torch.nn.functional as F
from .backbone import SiameseConvNeXtV2
from .modules import SC_UP_Module, BGI_Module, UWFF_Head

class BoundaryConditionedFusion(nn.Module):
    def __init__(self, in_channels, num_heads=8):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = in_channels // num_heads
        self.scale = self.head_dim ** -0.5

        self.norm_in = nn.BatchNorm2d(in_channels * 2)

        self.qkv_conv = nn.Conv2d(in_channels * 2, in_channels * 3, kernel_size=1, bias=False)
        
        self.boundary_proj = nn.Sequential(
            nn.Conv2d(1, in_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(in_channels),
            nn.Sigmoid()
        )
        
        self.proj = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.norm = nn.BatchNorm2d(in_channels)
        
        # Enhanced Local residual path with Depthwise Conv for high-freq structural preservation
        self.local_residual = nn.Sequential(
            nn.Conv2d(in_channels * 2, in_channels * 2, kernel_size=3, padding=1, groups=in_channels * 2),
            nn.Conv2d(in_channels * 2, in_channels, kernel_size=1),
            nn.BatchNorm2d(in_channels)
        )

    def forward(self, f1, f2, boundary_map):
        B, C, H, W = f1.shape
        
        b_down = F.interpolate(boundary_map, size=(H, W), mode='bilinear', align_corners=False)
        b_weight = self.boundary_proj(b_down) 
        
        x = torch.cat([f1, f2], dim=1) 
        x_norm = self.norm_in(x)
        
        qkv = self.qkv_conv(x_norm)   
        q, k, v = qkv.chunk(3, dim=1)  
        
        # Focus queries on boundary regions, and amplify retrieved values
        q = q * (1.0 + b_weight)
        v = v * (1.0 + b_weight)
        
        N = H * W
        q = q.view(B, self.num_heads, self.head_dim, N).transpose(-2, -1) 
        k = k.view(B, self.num_heads, self.head_dim, N)                   
        v = v.view(B, self.num_heads, self.head_dim, N).transpose(-2, -1) 
        
        q = q * self.scale
        
        attn = (q.float() @ k.float()) 
        attn = attn.softmax(dim=-1).to(v.dtype) 
        
        out = (attn @ v).transpose(-2, -1).reshape(B, C, H, W)
        
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
                x = F.interpolate(x, size=skip.shape[2:], mode='bilinear', align_corners=False)
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
        self.temporal_fusion = BoundaryConditionedFusion(in_channels=dims[3], num_heads=8)
        
        # [SOTA FIX 1] Multi-Scale Boundary Head. Combines Stage 0 and Stage 1
        # to ground the boundary map in semantics, preventing low-level noise interference.
        self.boundary_head = nn.Sequential(
            nn.Conv2d(dims[0] + dims[1], 128, 3, padding=1),
            nn.BatchNorm2d(128), nn.ReLU(inplace=True),
            nn.Conv2d(128, 64, 3, padding=1),
            nn.BatchNorm2d(64), nn.ReLU(inplace=True),
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
        
        # 1. Generate Multi-Scale Boundary Prior
        diff_0 = torch.abs(f1_list[0] - f2_list[0])
        diff_1 = torch.abs(f1_list[1] - f2_list[1])
        diff_1_up = F.interpolate(diff_1, size=diff_0.shape[2:], mode='bilinear', align_corners=False)
        
        boundary_logits = self.boundary_head(torch.cat([diff_0, diff_1_up], dim=1))
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
        
        out_ss1 = F.interpolate(out_ss1, size=img_size, mode='bilinear', align_corners=False)
        out_ss2 = F.interpolate(out_ss2, size=img_size, mode='bilinear', align_corners=False)
        out_cd = F.interpolate(out_cd, size=img_size, mode='bilinear', align_corners=False)
        out_bd = F.interpolate(boundary_logits, size=img_size, mode='bilinear', align_corners=False)
        
        return out_ss1, out_ss2, out_cd, out_bd
