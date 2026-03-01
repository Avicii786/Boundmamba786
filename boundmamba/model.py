import torch
import torch.nn as nn
import torch.nn.functional as F
from .backbone import SiameseConvNeXtV2
from .modules import SC_UP_Module, BGI_Module, UWFF_Head

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
        
        diff_low = torch.abs(f1_list[0] - f2_list[0])
        boundary_logits = self.boundary_head(diff_low)
        boundary_map = torch.sigmoid(boundary_logits)
        
        f1_3, f2_3 = self.sc_up(f1_list[3], f2_list[3])
        cd_3 = torch.abs(f1_3 - f2_3)
        
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
        
        # SOTA FIX: Raw logits are returned to prevent gradient starvation in dec_ss2
        return out_ss1, out_ss2, out_cd, out_bd
