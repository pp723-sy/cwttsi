import torch
from torch import nn
from torch import Tensor
from timm.models.layers import DropPath
from typing import List
import torch.nn.functional as F
import math


class LocalWindowSelfAttention(nn.Module):
    def __init__(self, dim: int, num_heads: int = 4, window_size: int = 7, attn_drop: float = 0.0,
                 proj_drop: float = 0.0):
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError('dim must be divisible by num_heads in LocalWindowSelfAttention.')
        self.dim = dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=True)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x: Tensor) -> Tensor:
        batch_size, channels, height, width = x.shape
        ws = self.window_size
        pad_h = (ws - height % ws) % ws
        pad_w = (ws - width % ws) % ws
        if pad_h > 0 or pad_w > 0:
            x = F.pad(x, (0, pad_w, 0, pad_h))
        _, _, h_pad, w_pad = x.shape
        num_h = h_pad // ws
        num_w = w_pad // ws

        x = x.permute(0, 2, 3, 1).contiguous()
        x = x.view(batch_size, num_h, ws, num_w, ws, channels)
        x = x.permute(0, 1, 3, 2, 4, 5).contiguous()
        windows = x.view(batch_size * num_h * num_w, ws * ws, channels)

        qkv = self.qkv(windows)
        qkv = qkv.view(qkv.size(0), qkv.size(1), 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        query, key, value = qkv[0], qkv[1], qkv[2]
        attention = (query @ key.transpose(-2, -1)) * self.scale
        attention = F.softmax(attention, dim=-1)
        attention = self.attn_drop(attention)
        out = attention @ value
        out = out.transpose(1, 2).contiguous().view(windows.size(0), ws * ws, channels)
        out = self.proj_drop(self.proj(out))

        out = out.view(batch_size, num_h, num_w, ws, ws, channels)
        out = out.permute(0, 1, 3, 2, 4, 5).contiguous()
        out = out.view(batch_size, h_pad, w_pad, channels)
        out = out.permute(0, 3, 1, 2).contiguous()

        if pad_h > 0 or pad_w > 0:
            out = out[:, :, :height, :width]
        return out


class HaloLocalAttentionBlock(nn.Module):
    def __init__(self, dim: int, num_heads: int = 4, window_size: int = 7):
        super().__init__()
        self.norm = nn.GroupNorm(num_groups=1, num_channels=dim)
        self.attn = LocalWindowSelfAttention(dim=dim, num_heads=num_heads, window_size=window_size)

    def forward(self, x: Tensor) -> Tensor:
        return x + self.attn(self.norm(x))


class ConvNeXtBlock(nn.Module):
    def __init__(self, in_features: int, out_features: int, expansion=4, layer_scaler_init_value=1e-6, drop_p=.0):
        super().__init__()
        expanded_features = out_features * expansion
        self.block = nn.Sequential(
            nn.Conv2d(in_features, in_features, kernel_size=7, padding=3, bias=False, groups=in_features),
            nn.GroupNorm(num_groups=1, num_channels=in_features),
            nn.Conv2d(in_features, expanded_features, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(expanded_features, out_features, kernel_size=1)
        )
        self.gamma = nn.Parameter(layer_scaler_init_value * torch.ones(out_features),
                                  requires_grad=True) if layer_scaler_init_value > 0 else None
        self.drop_path = DropPath(drop_p) if drop_p > 0. else nn.Identity()

    def forward(self, x: Tensor) -> Tensor:
        res = x
        x = self.block(x)
        if self.gamma is not None:
            x = self.gamma[None, ..., None, None] * x
        x = self.drop_path(x)
        x += res
        return x


class ConvNeXtStem(nn.Sequential):
    def __init__(self, in_features: int, stem_features: int):
        super().__init__(
            nn.Sequential(
                nn.Conv2d(in_features, stem_features, kernel_size=4, stride=4),
                nn.GroupNorm(num_groups=1, num_channels=stem_features)
            )
        )


class ConvNeXtEncoder(nn.Module):
    def __init__(self, in_features: int, depths: List[int], widths: List[int], drop_p=.0):
        super().__init__()
        self.stem = ConvNeXtStem(in_features, widths[0])
        drop_probs = [x.item() for x in torch.linspace(0, drop_p, sum(depths))]
        self.stages = nn.ModuleList()
        cur = 0
        for i in range(3):
            stage = nn.Sequential(
                *[ConvNeXtBlock(widths[i], widths[i], drop_p=drop_probs[cur + j]) for j in range(depths[i])],
                nn.GroupNorm(num_groups=1, num_channels=widths[i]),
                nn.Conv2d(widths[i], widths[i + 1], kernel_size=2, stride=2)
            )
            self.stages.append(stage)
            cur += depths[i]
        stage = nn.Sequential(
            *[ConvNeXtBlock(widths[3], widths[3], drop_p=drop_probs[cur + j]) for j in range(depths[3])]
        )
        self.stages.append(stage)

    def forward(self, x):
        x = self.stem(x)
        for stage in self.stages:
            x = stage(x)
        return x


class ConvNeXtHead(nn.Module):
    def __init__(self, in_features: int, num_nodes: int, predict_length: int):
        super().__init__()
        self.pool = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.LayerNorm(in_features)
        )
        self.fc = nn.Linear(in_features, num_nodes * predict_length)
        self.num_nodes = num_nodes
        self.predict_length = predict_length

    def forward(self, x):
        x = self.pool(x)
        x = self.fc(x)
        x = x.view(x.size(0), self.num_nodes, self.predict_length)
        return x


class ConvNeXt(nn.Sequential):
    def __init__(self, image_channels: int, num_nodes: int, depths: List, widths: List, predict_length: int,
                 halo_local_enabled: bool = False, halo_window_size: int = 7, halo_num_heads: int = 4):
        super().__init__()
        self.encoder = ConvNeXtEncoder(image_channels, depths, widths)
        self.halo_local_enabled = halo_local_enabled
        if self.halo_local_enabled:
            self.halo_block = HaloLocalAttentionBlock(
                dim=widths[-1],
                num_heads=halo_num_heads,
                window_size=halo_window_size,
            )
        self.head = ConvNeXtHead(widths[-1], num_nodes, predict_length)

    def forward(self, x):
        x = self.encoder(x)
        if self.halo_local_enabled:
            x = self.halo_block(x)
        x = self.head(x)
        return x


device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


















class STBAN(nn.Module):
    def __init__(self, num_of_vertices: int, num_his: int, num_pred: int, time_slots_per_day: int = 288,
                 bn_decay=0.1, set_dim: int = 3):
        super(STBAN, self).__init__()
        from external_sstban import load_components
        globals().update(load_components())
        gpu = 1
        L = 3
        K = 16
        d = 8
        in_channels = 1
        out_channels = 1
        self.L = L
        self.K = K
        self.d = d
        D = K * d
        self.set_dim = set_dim
        self.num_his = num_his
        self.input_dim = int(time_slots_per_day) + 7
        self.num_pred = num_pred
        self.num_of_vertices = num_of_vertices
        self.SE = nn.Parameter(torch.FloatTensor(self.num_of_vertices, D))
        nn.init.xavier_uniform_(self.SE)
        self.STEmbedding = STEmbedding(self.input_dim, D, bn_decay, gpu)
        self.STAttBlock_1 = nn.ModuleList(
            [STAttBlock(K, d, self.num_his, self.num_of_vertices, self.set_dim, bn_decay) for _ in range(L)]
        )
        self.STAttBlock_2 = nn.ModuleList(
            [STAttBlock(K, d, self.num_pred, self.num_of_vertices, self.set_dim, bn_decay) for _ in range(L)]
        )
        self.transformAttention = transformAttention(K, d, bn_decay)
        self.FC_1 = FC(input_dims=[in_channels, D], units=[D, D], activations=[F.relu, None], bn_decay=bn_decay)
        self.FC_2 = FC(input_dims=[D, D], units=[D, out_channels], activations=[F.relu, None], bn_decay=bn_decay)

    def forward(self, X, TE):
        X = self.FC_1(X)
        STE = self.STEmbedding(self.SE, TE, self.input_dim - 7)
        STE_his = STE[:, :self.num_his]
        STE_pred = STE[:, self.num_his:]
        for net in self.STAttBlock_1:
            X = net(X, STE_his)
        X = self.transformAttention(X, STE_his, STE_pred)
        for net in self.STAttBlock_2:
            X = net(X, STE_pred)
        pred = self.FC_2(X)
        pred = pred.flatten(2)
        pred = torch.permute(pred, (0, 2, 1))
        return pred


class ConvNeXtSTBAN(nn.Module):
    def __init__(self, num_nodes: int = 1, depths=None, widths=None, predict_length: int = 1,
                 history_step: int = 12, image_channels: int = 1, time_slots_per_day: int = 288,
                 fusion_init_conv: float = 0.0, fusion_init_sta: float = 0.0,
                 local_focus_enabled: bool = False, local_focus_kernel: int = 3,
                 local_focus_temperature: float = 1.0, local_focus_topk: int = 0,
                 halo_local_enabled: bool = False, halo_window_size: int = 7, halo_num_heads: int = 4,
                 route_set_dim: int = 3, ablation_mode: str = 'full'):
        super().__init__()
        valid_ablation_modes = {'full', 'no_img', 'no_ts', 'no_fuse'}
        if ablation_mode not in valid_ablation_modes:
            raise ValueError(f'ablation_mode must be one of {sorted(valid_ablation_modes)}.')
        self.ablation_mode = ablation_mode
        if widths is None:
            widths = [96, 192, 384, 768]
        if depths is None:
            depths = [3, 3, 9, 3]
        self.conv = None
        if self.ablation_mode != 'no_img':
            self.conv = ConvNeXt(
                image_channels,
                num_nodes,
                depths,
                widths,
                predict_length,
                halo_local_enabled=halo_local_enabled,
                halo_window_size=halo_window_size,
                halo_num_heads=halo_num_heads,
            )
        self.sta = None
        if self.ablation_mode != 'no_ts':
            self.sta = STBAN(
                num_of_vertices=num_nodes,
                num_his=history_step,
                num_pred=predict_length,
                time_slots_per_day=time_slots_per_day,
                set_dim=route_set_dim,
            )
        self.local_focus_enabled = local_focus_enabled
        self.local_focus_temperature = local_focus_temperature
        self.local_focus_topk = local_focus_topk
        if self.local_focus_enabled:
            if local_focus_kernel % 2 == 0:
                raise ValueError('local_focus_kernel must be odd.')
            self.local_focus_conv = nn.Conv1d(
                in_channels=1,
                out_channels=1,
                kernel_size=local_focus_kernel,
                padding=local_focus_kernel // 2,
                bias=True,
            )
        self.weight = nn.Parameter(torch.tensor([fusion_init_conv, fusion_init_sta], dtype=torch.float32))

    def apply_local_focus(self, x_ts):
        batch_size, num_nodes, history_step = x_ts.shape
        x_view = x_ts.reshape(batch_size * num_nodes, 1, history_step)
        logits = self.local_focus_conv(x_view) / max(self.local_focus_temperature, 1e-6)
        if self.local_focus_topk > 0 and self.local_focus_topk < history_step:
            topk_idx = torch.topk(logits, k=self.local_focus_topk, dim=-1).indices
            mask = torch.zeros_like(logits)
            mask.scatter_(-1, topk_idx, 1.0)
            logits = logits.masked_fill(mask == 0, -1e9)
        attention = F.softmax(logits, dim=-1)
        focused = x_view * (1.0 + attention)
        return focused.reshape(batch_size, num_nodes, history_step)

    def predict_sta(self, x_ts, te, mean, std):
        if self.local_focus_enabled:
            x_ts = self.apply_local_focus(x_ts)
        x_ts = x_ts.permute(0, 2, 1)
        x_ts = x_ts.view(x_ts.size(0), x_ts.size(1), x_ts.size(2), 1)
        pred_sta = self.sta(x_ts, te)
        pred_sta = pred_sta * std + mean
        return torch.nan_to_num(pred_sta, nan=0.0, posinf=1e6, neginf=-1e6)

    def forward(self, x_image, x_ts, te, mean, std):
        if self.ablation_mode == 'no_img':
            return self.predict_sta(x_ts, te, mean, std)

        pred_conv = self.conv(x_image)
        pred_conv = torch.nan_to_num(pred_conv, nan=0.0, posinf=1e6, neginf=-1e6)

        if self.ablation_mode == 'no_ts':
            return pred_conv

        pred_sta = self.predict_sta(x_ts, te, mean, std)
        if self.ablation_mode == 'no_fuse':
            pred = 0.5 * pred_conv + 0.5 * pred_sta
            return torch.nan_to_num(pred, nan=0.0, posinf=1e6, neginf=-1e6)

        weights = torch.softmax(self.weight, dim=0)
        w1 = weights[0]
        w2 = weights[1]
        pred = pred_conv * w1 + pred_sta * w2
        return torch.nan_to_num(pred, nan=0.0, posinf=1e6, neginf=-1e6)


def __getattr__(name):
    from external_sstban import NAMES, load_components
    if name in NAMES:
        return load_components()[name]
    raise AttributeError(name)
