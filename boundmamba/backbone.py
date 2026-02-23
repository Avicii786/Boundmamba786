import os
import math
import copy
from functools import partial
from typing import Optional, Callable, Any
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
from einops import rearrange, repeat
from timm.models.layers import DropPath, trunc_normal_, to_2tuple

# Import generic selective scan from the installed mamba_ssm
try:
    from mamba_ssm.ops.selective_scan_interface import selective_scan_fn
except ImportError:
    raise ImportError("mamba_ssm is not installed. Please install it via `pip install mamba_ssm`")

# ==============================================================================
# 1. Helper Layers
# ==============================================================================
class LayerNorm2d(nn.Module):
    def __init__(self, normalized_shape, eps=1e-6, elementwise_affine=True):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape)) if elementwise_affine else None
        self.bias = nn.Parameter(torch.zeros(normalized_shape)) if elementwise_affine else None
        self.eps = eps
        self.normalized_shape = (normalized_shape, )

    def forward(self, x):
        return F.layer_norm(x.permute(0, 2, 3, 1), self.normalized_shape, self.weight, self.bias, self.eps).permute(0, 3, 1, 2)

class Permute(nn.Module):
    def __init__(self, *args):
        super().__init__()
        self.args = args
    def forward(self, x):
        return x.permute(*self.args)

# ==============================================================================
# 2. Cross Scan & Merge (Exact Logic from vmamba.py)
# ==============================================================================
class CrossScan(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor):
        B, C, H, W = x.shape
        ctx.shape = (B, C, H, W)
        xs = x.new_empty((B, 4, C, H * W))
        xs[:, 0] = x.flatten(2, 3)
        xs[:, 1] = x.transpose(dim0=2, dim1=3).flatten(2, 3)
        xs[:, 2:4] = torch.flip(xs[:, 0:2], dims=[-1])
        return xs
    
    @staticmethod
    def backward(ctx, ys: torch.Tensor):
        B, C, H, W = ctx.shape
        L = H * W
        ys = ys[:, 0:2] + torch.flip(ys[:, 2:4], dims=[-1])
        y = ys[:, 0] + ys[:, 1].view(B, -1, W, H).transpose(dim0=2, dim1=3).contiguous().view(B, -1, L)
        return y.view(B, C, H, W)

class CrossMerge(torch.autograd.Function):
    @staticmethod
    def forward(ctx, ys: torch.Tensor):
        B, K, D, H, W = ys.shape
        ctx.shape = (H, W)
        ys = ys.view(B, K, D, -1)
        ys = ys[:, 0:2] + torch.flip(ys[:, 2:4], dims=[-1])
        y = ys[:, 0] + ys[:, 1].view(B, -1, W, H).transpose(dim0=2, dim1=3).contiguous().view(B, -1, H * W)
        return y.view(B, D, H, W)
    
    @staticmethod
    def backward(ctx, x: torch.Tensor):
        H, W = ctx.shape
        B, C, L = x.shape
        x = x.view(B, C, H, W)
        xs = x.new_empty((B, 4, C, L))
        xs[:, 0] = x.flatten(2, 3)
        xs[:, 1] = x.transpose(dim0=2, dim1=3).flatten(2, 3)
        xs[:, 2:4] = torch.flip(xs[:, 0:2], dims=[-1])
        return xs.view(B, 4, C, H, W)

# ==============================================================================
# 3. SS2D: The Core VMamba Block (Using mamba_ssm)
# ==============================================================================
class SS2D(nn.Module):
    def __init__(self, d_model, d_state=16, d_conv=3, expand=2, dt_rank="auto", 
                 dt_min=0.001, dt_max=0.1, dt_init="random", dt_scale=1.0, 
                 dt_init_floor=1e-4, dropout=0., conv_bias=True, bias=False, device=None, dtype=None):
        super().__init__()
        factory_kwargs = {"device": device, "dtype": dtype}
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.d_inner = int(self.expand * self.d_model)
        self.dt_rank = math.ceil(self.d_model / 16) if dt_rank == "auto" else dt_rank

        self.in_proj = nn.Linear(self.d_model, self.d_inner * 2, bias=bias, **factory_kwargs)
        self.conv2d = nn.Conv2d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            groups=self.d_inner,
            bias=conv_bias,
            kernel_size=d_conv,
            padding=(d_conv - 1) // 2,
            **factory_kwargs,
        )
        self.act = nn.SiLU()

        self.x_proj = (
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs),
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs),
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs),
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs),
        )
        self.x_proj_weight = nn.Parameter(torch.stack([t.weight for t in self.x_proj], dim=0)) # (K=4, N, D)
        del self.x_proj

        self.dt_projs = (
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor, **factory_kwargs),
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor, **factory_kwargs),
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor, **factory_kwargs),
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor, **factory_kwargs),
        )
        self.dt_projs_weight = nn.Parameter(torch.stack([t.weight for t in self.dt_projs], dim=0)) # (K=4, D, N)
        self.dt_projs_bias = nn.Parameter(torch.stack([t.bias for t in self.dt_projs], dim=0)) # (K=4, D)
        del self.dt_projs

        self.A_logs = self.A_log_init(self.d_state, self.d_inner, copies=4, merge=True) # (K=4, D, N)
        self.Ds = self.D_init(self.d_inner, copies=4, merge=True) # (K=4, D, N)

        self.out_norm = nn.LayerNorm(self.d_inner)
        self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=bias, **factory_kwargs)
        self.dropout = nn.Dropout(dropout) if dropout > 0. else None

    @staticmethod
    def dt_init(dt_rank, d_inner, dt_scale=1.0, dt_init="random", dt_min=0.001, dt_max=0.1, dt_init_floor=1e-4, **factory_kwargs):
        proj = nn.Linear(dt_rank, d_inner, bias=True, **factory_kwargs)
        dt_init_std = dt_rank**-0.5 * dt_scale
        if dt_init == "constant":
            nn.init.constant_(proj.weight, dt_init_std)
        elif dt_init == "random":
            nn.init.uniform_(proj.weight, -dt_init_std, dt_init_std)
        else:
            raise NotImplementedError
        
        dt = torch.exp(
            torch.rand(d_inner, **factory_kwargs) * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        ).clamp(min=dt_init_floor)
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            proj.bias.copy_(inv_dt)
        proj.bias.requires_grad = False
        return proj

    @staticmethod
    def A_log_init(d_state, d_inner, copies=1, device=None, merge=True):
        A = repeat(
            torch.arange(1, d_state + 1, dtype=torch.float32, device=device),
            "n -> d n", d=d_inner,
        ).contiguous()
        A_log = torch.log(A)
        if copies > 1:
            A_log = repeat(A_log, "d n -> r d n", r=copies)
            if merge:
                A_log = A_log.flatten(0, 1)
        return nn.Parameter(A_log).contiguous()

    @staticmethod
    def D_init(d_inner, copies=1, device=None, merge=True):
        D = torch.ones(d_inner, device=device)
        if copies > 1:
            D = repeat(D, "n1 -> r n1", r=copies)
            if merge:
                D = D.flatten(0, 1)
        return nn.Parameter(D).contiguous()

    def forward(self, x):
        B, H, W, C = x.shape
        xz = self.in_proj(x)
        x, z = xz.chunk(2, dim=-1)
        
        # Conv2d path
        x = x.permute(0, 3, 1, 2).contiguous()
        x = self.act(self.conv2d(x))
        x = self.act(x)
        
        # Cross Scan
        x = CrossScan.apply(x) # (B, 4, C, H*W)
        
        # SSM parameters projection
        x_dbl = torch.einsum("b k d l, k c d -> b k c l", x.view(B, 4, -1, H*W), self.x_proj_weight)
        dts, Bs, Cs = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=2)
        dts = torch.einsum("b k r l, k d r -> b k d l", dts.view(B, 4, -1, H*W), self.dt_projs_weight)
        
        xs = x.float().view(B, -1, H*W)
        dts = dts.contiguous().float().view(B, -1, H*W)
        Bs = Bs.float().view(B, 4, -1, H*W)
        Cs = Cs.float().view(B, 4, -1, H*W)
        Ds = self.Ds.float().view(-1)
        As = -torch.exp(self.A_logs.float()).view(-1, self.d_state)
        dt_projs_bias = self.dt_projs_bias.float().view(-1)

        # Selective Scan Execution (Process 4 directions)
        out_y = []
        for i in range(4):
            # FIXED: Removed .permute(0, 2, 1) from inputs to match mamba_ssm shape (B, D, L)
            yi = selective_scan_fn(
                xs.view(B, 4, -1, H*W)[:, i],
                dts.view(B, 4, -1, H*W)[:, i],
                As[i*self.d_inner:(i+1)*self.d_inner],
                Bs[:, i],
                Cs[:, i],
                Ds[i*self.d_inner:(i+1)*self.d_inner],
                delta_bias=dt_projs_bias[i*self.d_inner:(i+1)*self.d_inner],
                delta_softplus=True
            )
            out_y.append(yi)
            
        y = torch.stack(out_y, dim=1) 
        # FIXED: Since inputs are (B, D, L), yi is (B, D, L), y is (B, 4, D, L). No permutation needed.
        y = y.contiguous().view(B, 4, -1, H, W)
        
        # Cross Merge
        y = CrossMerge.apply(y) # (B, D, H, W)
        y = y.permute(0, 2, 3, 1) # (B, H, W, D)

        y = self.out_norm(y)
        y = y * F.silu(z)
        out = self.out_proj(y)
        if self.dropout is not None:
            out = self.dropout(out)
        return out

# ==============================================================================
# 4. VSS Blocks & Layers
# ==============================================================================
class VSSBlock(nn.Module):
    def __init__(self, hidden_dim: int = 0, drop_path: float = 0, norm_layer: Callable[..., torch.nn.Module] = partial(nn.LayerNorm, eps=1e-6), attn_drop_rate: float = 0, d_state: int = 16, **kwargs):
        super().__init__()
        self.ln_1 = norm_layer(hidden_dim)
        self.self_attention = SS2D(d_model=hidden_dim, d_state=d_state, dropout=attn_drop_rate, **kwargs)
        self.drop_path = DropPath(drop_path)
        self.ln_2 = norm_layer(hidden_dim)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.GELU(),
            nn.Linear(hidden_dim * 4, hidden_dim)
        )

    def forward(self, input: torch.Tensor):
        x = input + self.drop_path(self.self_attention(self.ln_1(input)))
        x = x + self.drop_path(self.mlp(self.ln_2(x)))
        return x

class VSSLayer(nn.Module):
    def __init__(self, dim, depth, attn_drop=0., drop_path=0., norm_layer=nn.LayerNorm, downsample=None, d_state=16):
        super().__init__()
        self.blocks = nn.ModuleList([
            VSSBlock(hidden_dim=dim, drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path, norm_layer=norm_layer, attn_drop_rate=attn_drop, d_state=d_state)
            for i in range(depth)
        ])
        if downsample is not None:
            self.downsample = downsample(dim=dim, norm_layer=norm_layer)
        else:
            self.downsample = None

    def forward(self, x):
        for blk in self.blocks:
            x = blk(x)
        if self.downsample is None:
            return x, x
        y = self.downsample(x)
        return x, y # Return (processed, downsampled)

class PatchMerging2D(nn.Module):
    def __init__(self, dim, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)
        self.norm = norm_layer(4 * dim)

    def forward(self, x):
        B, H, W, C = x.shape
        x0 = x[:, 0::2, 0::2, :]
        x1 = x[:, 1::2, 0::2, :]
        x2 = x[:, 0::2, 1::2, :]
        x3 = x[:, 1::2, 1::2, :]
        x = torch.cat([x0, x1, x2, x3], -1)
        x = self.norm(x)
        x = self.reduction(x)
        return x

# ==============================================================================
# 5. Main VSSM Backbone (Merged VSSM + Backbone_VSSM Wrapper)
# ==============================================================================
class Backbone_VSSM(nn.Module):
    def __init__(self, in_chans=3, patch_size=4, dims=[96, 192, 384, 768], depths=[2, 2, 9, 2], 
                 d_state=16, drop_rate=0., attn_drop_rate=0., drop_path_rate=0.2, 
                 out_indices=(0, 1, 2, 3), pretrained=None, norm_layer='ln2d'):
        super().__init__()
        self.dims = dims
        self.out_indices = out_indices
        
        # Patch Embedding
        self.patch_embed = nn.Sequential(
            nn.Conv2d(in_chans, dims[0], kernel_size=patch_size, stride=patch_size),
            Permute(0, 2, 3, 1),
            nn.LayerNorm(dims[0])
        )

        # Build Layers
        self.layers = nn.ModuleList()
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        
        for i_layer in range(len(depths)):
            is_last = (i_layer == len(depths) - 1)
            layer = VSSLayer(
                dim=dims[i_layer],
                depth=depths[i_layer],
                d_state=d_state,
                drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],
                norm_layer=nn.LayerNorm,
                downsample=PatchMerging2D if not is_last else None
            )
            self.layers.append(layer)

        # Output Norm Selection
        _NORMLAYERS = dict(
            ln=nn.LayerNorm,
            ln2d=LayerNorm2d,
            bn=nn.BatchNorm2d,
        )
        norm_layer_cls = _NORMLAYERS.get(norm_layer.lower(), LayerNorm2d)
        self.channel_first = (norm_layer.lower() in ["bn", "ln2d"])

        for i in out_indices:
            layer = norm_layer_cls(self.dims[i])
            layer_name = f'outnorm{i}'
            self.add_module(layer_name, layer)

        self.apply(self._init_weights)
        if pretrained:
            self.load_pretrained(pretrained)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def load_pretrained(self, ckpt=None, key="model"):
        if ckpt is None: return
        try:
            _ckpt = torch.load(ckpt, map_location='cpu')
            if 'model' in _ckpt:
                state_dict = _ckpt['model']
            else:
                state_dict = _ckpt
            
            # Remove keys that might conflict or handle prefixes
            new_dict = OrderedDict()
            for k, v in state_dict.items():
                if k.startswith('backbone.'):
                    new_dict[k[9:]] = v
                elif k.startswith('classifier'):
                    continue # Skip classifier weights
                else:
                    new_dict[k] = v
                    
            msg = self.load_state_dict(new_dict, strict=False)
            print(f"Loaded VMamba weights. Msg: {msg}")
        except Exception as e:
            print(f"Warning: Failed to load VMamba weights: {e}")

    def forward(self, x):
        # x: [B, 3, H, W]
        x = self.patch_embed(x) # [B, H/4, W/4, C]
        outs = []
        
        for i, layer in enumerate(self.layers):
            # layer.forward returns (current_stage_out, downsampled_out)
            o, x = layer(x) 
            
            if i in self.out_indices:
                norm_layer = getattr(self, f'outnorm{i}')
                # Apply norm. Check if norm requires channel first (BCHW)
                if self.channel_first:
                     # o is BHWC from VSSLayer, convert to BCHW
                    o_norm = norm_layer(o.permute(0, 3, 1, 2))
                else:
                    o_norm = norm_layer(o)
                    o_norm = o_norm.permute(0, 3, 1, 2)
                    
                outs.append(o_norm)
                
        return outs
