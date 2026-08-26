"""
GridDehazeNet — CORRECTED implementation matching outdoor_haze_best_3_6__1_

TWO BUGS FIXED:
  Bug 1 — Wrong activation: LeakyReLU -> ReLU inside DenseLayer
  Bug 2 — Missing residual scale: added 0.2 factor in RDB forward pass
These two together caused feature explosion (values reaching +-6000)
producing the yellow/saturated output seen in the screenshots.
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F


class DenseLayer(nn.Module):
    """FIX 1: Uses ReLU not LeakyReLU"""
    def __init__(self, in_ch: int, growth: int = 16):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, growth, 3, 1, 1, bias=True)
    def forward(self, x):
        return F.relu(self.conv(x), inplace=True)   # <-- ReLU


class RDB(nn.Module):
    """FIX 2: residual scaled by 0.2"""
    def __init__(self, in_ch: int, growth: int = 16, num_layers: int = 4, res_scale: float = 0.2):
        super().__init__()
        self.res_scale = res_scale                  # <-- 0.2 scaling
        self.residual_dense_layers = nn.ModuleList()
        ch = in_ch
        for _ in range(num_layers):
            self.residual_dense_layers.append(DenseLayer(ch, growth))
            ch += growth
        self.conv_1x1 = nn.Conv2d(ch, in_ch, 1, 1, 0, bias=True)
    def forward(self, x):
        feats = [x]
        for l in self.residual_dense_layers:
            feats.append(l(torch.cat(feats, dim=1)))
        return x + self.res_scale * self.conv_1x1(torch.cat(feats, dim=1))


class DownsampleBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, in_ch,  3, stride=2, padding=1, bias=True)
        self.conv2 = nn.Conv2d(in_ch, out_ch, 3, stride=1, padding=1, bias=True)
    def forward(self, x):
        return F.relu(self.conv2(F.relu(self.conv1(x), inplace=True)), inplace=True)


class UpsampleBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.deconv = nn.ConvTranspose2d(in_ch, in_ch, 3, stride=2, padding=1, output_padding=1, bias=True)
        self.conv   = nn.Conv2d(in_ch, out_ch, 3, 1, 1, bias=True)
    def forward(self, x):
        return F.relu(self.conv(F.relu(self.deconv(x), inplace=True)), inplace=True)


class GridDehazeNet(nn.Module):
    """3-scale x 5-column RDB grid matching outdoor_haze_best_3_6__1_ checkpoint."""
    CH = [16, 32, 64]

    def __init__(self, res_scale: float = 0.2):
        super().__init__()
        C = self.CH
        self.conv_in  = nn.Conv2d(3,    C[0], 3, 1, 1, bias=True)
        self.conv_out = nn.Conv2d(C[0], 3,    3, 1, 1, bias=True)
        self.rdb_in   = RDB(C[0], res_scale=res_scale)
        self.rdb_out  = RDB(C[0], res_scale=res_scale)
        self.rdb_module = nn.ModuleDict({
            f'{s}_{col}': RDB(C[s], res_scale=res_scale)
            for s in range(3) for col in range(5)
        })
        self.downsample_module = nn.ModuleDict({
            f'{s}_{col}': DownsampleBlock(C[s], C[s+1])
            for s in range(2) for col in range(3)
        })
        self.upsample_module = nn.ModuleDict({
            f'{s}_{col}': UpsampleBlock(C[s+1], C[s])
            for s in range(2) for col in range(3, 6)
        })
        self.coefficient = nn.Parameter(torch.ones(3, 6, 2, 64) * 0.5)

    def _blend(self, a, b, scale, col):
        if a.shape[2:] != b.shape[2:]:
            b = F.interpolate(b, size=a.shape[2:], mode='bilinear', align_corners=False)
        ch = a.shape[1]
        w  = torch.softmax(self.coefficient[scale, col, :, :ch], dim=0)
        return w[0].view(1,ch,1,1)*a + w[1].view(1,ch,1,1)*b

    def forward(self, x):
        f = F.relu(self.conv_in(x), inplace=True)
        f = self.rdb_in(f)
        g = [[None]*5 for _ in range(3)]

        # Col 0
        g[0][0] = self.rdb_module['0_0'](f)
        g[1][0] = self.rdb_module['1_0'](self.downsample_module['0_0'](g[0][0]))
        g[2][0] = self.rdb_module['2_0'](self.downsample_module['1_0'](g[1][0]))

        # Cols 1,2 — down path
        for col in range(1, 3):
            g[0][col] = self.rdb_module[f'0_{col}'](g[0][col-1])
            g[1][col] = self.rdb_module[f'1_{col}'](
                self._blend(g[1][col-1], self.downsample_module[f'0_{col}'](g[0][col]), 1, col))
            g[2][col] = self.rdb_module[f'2_{col}'](
                self._blend(g[2][col-1], self.downsample_module[f'1_{col}'](g[1][col]), 2, col))

        # Col 3 — turn
        g[2][3] = self.rdb_module['2_3'](g[2][2])
        g[1][3] = self.rdb_module['1_3'](self._blend(g[1][2], self.upsample_module['1_3'](g[2][3]), 1, 3))
        g[0][3] = self.rdb_module['0_3'](self._blend(g[0][2], self.upsample_module['0_3'](g[1][3]), 0, 3))

        # Col 4 — up
        g[2][4] = self.rdb_module['2_4'](g[2][3])
        g[1][4] = self.rdb_module['1_4'](self._blend(g[1][3], self.upsample_module['1_4'](g[2][4]), 1, 4))
        g[0][4] = self.rdb_module['0_4'](self._blend(g[0][3], self.upsample_module['0_4'](g[1][4]), 0, 4))

        # Col 5 — upsample only (no RDB in checkpoint at col 5)
        us_2_1 = self.upsample_module['1_5'](g[2][4])
        bl_1   = self._blend(g[1][4], us_2_1, 1, 5)
        us_1_0 = self.upsample_module['0_5'](bl_1)
        final  = self._blend(g[0][4], us_1_0, 0, 5)

        out = self.rdb_out(final)
        out = self.conv_out(out)
        return torch.clamp(x + out, 0.0, 1.0)


def build_griddehazenet(weights_path: str = None, device: str = 'cpu') -> GridDehazeNet:
    """Load GridDehazeNet, stripping DataParallel 'module.' prefix automatically."""
    model = GridDehazeNet().to(device)
    if not weights_path or not os.path.exists(weights_path):
        print(f'[GridDehazeNet] Demo mode' + (f' — not found: {weights_path}' if weights_path else ''))
        model.eval(); return model

    raw = torch.load(weights_path, map_location=device, weights_only=False)
    if isinstance(raw, dict):
        if all(k.startswith('module.') for k in raw):
            state = {k[len('module.'):]: v for k,v in raw.items()}
        elif 'state_dict' in raw:
            sd = raw['state_dict']
            state = {k[len('module.'):]: v for k,v in sd.items()} if all(k.startswith('module.') for k in sd) else sd
        else:
            state = raw
    elif isinstance(raw, nn.Module):
        state = raw.state_dict()
    else:
        state = raw

    missing, unexpected = model.load_state_dict(state, strict=False)
    n_ok = len(state) - len(unexpected)
    print(f'[GridDehazeNet] {n_ok}/{len(state)} tensors loaded from {os.path.basename(weights_path)}'
          + (f'  missing:{len(missing)}' if missing else ''))
    model.eval(); return model