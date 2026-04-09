import os
import torch
import torch.nn as nn
import timm

try:
    from safetensors.torch import load_file as load_safetensors
except ImportError:
    load_safetensors = None

class TriDimensionalInteraction(nn.Module):
    """
    [NOVEL CONTRIBUTION] Tri-Dimensional Temporal Interaction (TDTI).
    Extracts Channel, Height, and Width attentions from the difference features
    and injects them back to guide the Siamese encoders.
    """
    def __init__(self, dim):
        super().__init__()
        reduction = 4
        mid_dim = max(16, dim // reduction)
        
        # 1. Channel Branch
        self.channel_gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(dim, mid_dim, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_dim, dim, kernel_size=1),
            nn.Sigmoid()
        )
        
        # 2. Height Branch (Strip Pooling Vertical)
        self.height_gate = nn.Sequential(
            nn.Conv2d(dim, mid_dim, kernel_size=1),
            nn.BatchNorm2d(mid_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_dim, dim, kernel_size=1),
            nn.Sigmoid()
        )
        
        # 3. Width Branch (Strip Pooling Horizontal)
        self.width_gate = nn.Sequential(
            nn.Conv2d(dim, mid_dim, kernel_size=1),
            nn.BatchNorm2d(mid_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_dim, dim, kernel_size=1),
            nn.Sigmoid()
        )
        
        # Learnable injection factor (Zero-Init preserves pretrained weights initially)
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x1, x2):
        # 1. Compute Raw Difference
        diff = torch.abs(x1 - x2)
        
        # 2. Attentions
        att_c = self.channel_gate(diff) 
        diff_h = diff.mean(dim=3, keepdim=True)
        att_h = self.height_gate(diff_h)
        diff_w = diff.mean(dim=2, keepdim=True)
        att_w = self.width_gate(diff_w)
        
        # 3. Combine Attentions (Fixing the Attention Collapse Bottleneck)
        # We average them instead of multiplying to prevent the gradients 
        # from exponentially vanishing when multiple sigmoids are chained.
        combined_att = (att_c + att_h + att_w) / 3.0
        
        # 4. Residual Injection
        x1_out = x1 + self.gamma * (x1 * combined_att)
        x2_out = x2 + self.gamma * (x2 * combined_att)
        
        return x1_out, x2_out

class SiameseConvNeXtV2(nn.Module):
    """
    Siamese ConvNeXt V2 with Tri-Dimensional Temporal Interaction.
    """
    def __init__(self, model_type='convnextv2_tiny', in_chans=3, checkpoint_path=None, drop_path_rate=0.2):
        super().__init__()
        
        print(f"[Backbone] Initializing {model_type} with TDTI Interaction...")
        
        # Determine whether to download automatically or use local weights
        download_pretrained = True if checkpoint_path is None else False
        if download_pretrained:
            print("[Backbone] No local weights provided. Will attempt to download from TIMM/HuggingFace Hub...")

        try:
            self.base_model = timm.create_model(
                model_type, 
                pretrained=download_pretrained,  # Automatically downloads if True
                in_chans=in_chans, 
                features_only=False, 
                num_classes=0,
                drop_path_rate=drop_path_rate
            )
        except Exception as e:
            print(f"[Backbone] Error creating model. Check internet connection or model_type name. Error: {e}")
            raise e

        # Local Loading Logic (Only triggers if a path is explicitly provided)
        if checkpoint_path and os.path.isfile(checkpoint_path):
            print(f"[Backbone] Loading local weights from {checkpoint_path}")
            try:
                if checkpoint_path.endswith('.safetensors') and load_safetensors is not None:
                    state_dict = load_safetensors(checkpoint_path)
                else:
                    state_dict = torch.load(checkpoint_path, map_location='cpu')

                if 'model' in state_dict:
                    state_dict = state_dict['model']
                
                missing, unexpected = self.base_model.load_state_dict(state_dict, strict=False)
                print(f"[Backbone] Local weights loaded successfully. Missing Keys: {len(missing)}")
            except Exception as e:
                print(f"[Backbone] Error loading local weights: {e}")
        
        # Set dimensions based on ConvNeXt architecture variants
        # Tiny and Small share the same channel dimensions, Base is wider
        if 'tiny' in model_type or 'small' in model_type: 
            self.dims = [96, 192, 384, 768]
        elif 'base' in model_type:
            self.dims = [128, 256, 512, 1024]
        elif 'large' in model_type:
            self.dims = [192, 384, 768, 1536]
        else: 
            self.dims = [96, 192, 384, 768] # Default
        
        print(f"[Backbone] Encoded Feature Dims: {self.dims}")

        self.stem = self.base_model.stem
        self.stages = self.base_model.stages
        
        # Insert Tri-Dimensional Interaction after each stage
        self.interactions = nn.ModuleList([
            TriDimensionalInteraction(dim) for dim in self.dims
        ])

    def forward(self, x1, x2):
        f1 = self.stem(x1)
        f2 = self.stem(x2)
        
        features_1 = []
        features_2 = []
        
        for i, stage in enumerate(self.stages):
            f1 = stage(f1)
            f2 = stage(f2)
            
            # TDTI Interaction
            f1, f2 = self.interactions[i](f1, f2)
            
            features_1.append(f1)
            features_2.append(f2)
            
        return features_1, features_2
