import torch
import torch.nn as nn
import torch.nn.functional as F

class ASPP(nn.Module):
    """ Atrous Spatial Pyramid Pooling for Global Context """
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 1, bias=False)
        self.conv2 = nn.Conv2d(in_channels, out_channels, 3, padding=6, dilation=6, bias=False)
        self.conv3 = nn.Conv2d(in_channels, out_channels, 3, padding=12, dilation=12, bias=False)
        self.conv4 = nn.Conv2d(in_channels, out_channels, 3, padding=18, dilation=18, bias=False)
        self.global_avg_pool = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Conv2d(in_channels, out_channels, 1, bias=False)
        )
        self.out_conv = nn.Sequential(
            nn.Conv2d(out_channels * 5, out_channels, 1, bias=False),
            nn.GroupNorm(16, out_channels), # [SOTA BATCH FIX] GroupNorm instead of BatchNorm
            nn.ReLU(inplace=True),
            nn.Dropout(0.1)
        )

    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.conv2(x)
        x3 = self.conv3(x)
        x4 = self.conv4(x)
        x5 = self.global_avg_pool(x)
        x5 = F.interpolate(x5, size=x.shape[2:], mode='bilinear', align_corners=False)
        x_concat = torch.cat([x1, x2, x3, x4, x5], dim=1)
        return self.out_conv(x_concat)

class SC_UP_Module(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        mid_ch = in_channels // 2
        self.aspp1 = ASPP(in_channels, mid_ch)
        self.aspp2 = ASPP(in_channels, mid_ch)
        
        self.corr_conv = nn.Sequential(
            nn.Conv2d(mid_ch * 2, mid_ch, 3, padding=1),
            nn.GroupNorm(16, mid_ch), # [SOTA BATCH FIX]
            nn.ReLU(inplace=True)
        )
        self.proj1 = nn.Conv2d(mid_ch, in_channels, 1)
        self.proj2 = nn.Conv2d(mid_ch, in_channels, 1)

    def forward(self, f1, f2):
        f1_ctx = self.aspp1(f1)
        f2_ctx = self.aspp2(f2)
        
        concat = torch.cat([f1_ctx, f2_ctx], dim=1)
        corr = self.corr_conv(concat)
        
        f1_out = f1 + self.proj1(f1_ctx * corr)
        f2_out = f2 + self.proj2(f2_ctx * corr)
        
        return f1_out, f2_out

class BGI_Module(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.spatial_gate = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1),
            nn.GroupNorm(8, 16), # [SOTA BATCH FIX]
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 1, 1),
            nn.Sigmoid()
        )
        self.cd_fusion = nn.Sequential(
            nn.Conv2d(in_channels * 3, in_channels, 3, padding=1),
            nn.GroupNorm(16, in_channels), # [SOTA BATCH FIX]
            nn.ReLU(inplace=True)
        )
        self.ss1_fusion = nn.Sequential(
            nn.Conv2d(in_channels * 2, in_channels, 3, padding=1),
            nn.GroupNorm(16, in_channels), # [SOTA BATCH FIX]
            nn.ReLU(inplace=True)
        )
        self.ss2_fusion = nn.Sequential(
            nn.Conv2d(in_channels * 2, in_channels, 3, padding=1),
            nn.GroupNorm(16, in_channels), # [SOTA BATCH FIX]
            nn.ReLU(inplace=True)
        )

    def forward(self, x1, x2, cd, bd_map):
        if bd_map.shape[2:] != cd.shape[2:]:
            bd_map = F.interpolate(bd_map, size=cd.shape[2:], mode='bilinear', align_corners=False)
        
        gate = self.spatial_gate(bd_map)
        cd_sharpened = cd * (1 + gate)
        
        cd_out = self.cd_fusion(torch.cat([x1, x2, cd_sharpened], dim=1))
        x1_out = self.ss1_fusion(torch.cat([x1, cd_out], dim=1))
        x2_out = self.ss2_fusion(torch.cat([x2, cd_out], dim=1))
        
        return x1_out, x2_out, cd_out

class UWFF_Head(nn.Module):
    def __init__(self, in_channels, num_classes):
        super().__init__()
        self.ss1_head = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 3, padding=1),
            nn.GroupNorm(16, in_channels), # [SOTA BATCH FIX]
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, num_classes, 1)
        )
        self.ss2_head = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 3, padding=1),
            nn.GroupNorm(16, in_channels), # [SOTA BATCH FIX]
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, num_classes, 1)
        )
        self.cd_head = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 3, padding=1),
            nn.GroupNorm(16, in_channels), # [SOTA BATCH FIX]
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, 1, 1)
        )

    def forward(self, x1, x2, cd_f):
        return self.ss1_head(x1), self.ss2_head(x2), self.cd_head(cd_f)
