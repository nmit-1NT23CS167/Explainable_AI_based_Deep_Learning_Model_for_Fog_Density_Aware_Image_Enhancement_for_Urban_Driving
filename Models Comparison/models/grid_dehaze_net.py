"""GridDehazeNet implementation compatible with the bundled RDB checkpoint."""

import os

import torch
import torch.nn as nn
import torch.nn.functional as F


class DenseLayer(nn.Module):
    def __init__(self, in_ch: int, growth: int = 16):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, growth, 3, 1, 1, bias=True)

    def forward(self, x):
        return F.relu(self.conv(x), inplace=True)


class RDB(nn.Module):
    def __init__(self, in_ch: int, growth: int = 16, num_layers: int = 4,
                 res_scale: float = 0.2):
        super().__init__()
        self.res_scale = res_scale
        layers = []
        channels = in_ch
        for _ in range(num_layers):
            layers.append(DenseLayer(channels, growth))
            channels += growth
        self.residual_dense_layers = nn.ModuleList(layers)
        self.conv_1x1 = nn.Conv2d(channels, in_ch, 1, bias=True)

    def forward(self, x):
        features = [x]
        for layer in self.residual_dense_layers:
            features.append(layer(torch.cat(features, dim=1)))
        return x + self.res_scale * self.conv_1x1(torch.cat(features, dim=1))


class DownsampleBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, in_ch, 3, stride=2, padding=1)
        self.conv2 = nn.Conv2d(in_ch, out_ch, 3, padding=1)

    def forward(self, x):
        return F.relu(self.conv2(F.relu(self.conv1(x), inplace=True)), inplace=True)


class UpsampleBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.deconv = nn.ConvTranspose2d(
            in_ch, in_ch, 3, stride=2, padding=1, output_padding=1)
        self.conv = nn.Conv2d(in_ch, out_ch, 3, padding=1)

    def forward(self, x):
        return F.relu(self.conv(F.relu(self.deconv(x), inplace=True)), inplace=True)


class GridDehazeNet(nn.Module):
    """Three-scale, five-column RDB grid used by the supplied checkpoint."""

    CHANNELS = (16, 32, 64)

    def __init__(self, res_scale: float = 0.2):
        super().__init__()
        channels = self.CHANNELS
        self.conv_in = nn.Conv2d(3, channels[0], 3, padding=1)
        self.conv_out = nn.Conv2d(channels[0], 3, 3, padding=1)
        self.rdb_in = RDB(channels[0], res_scale=res_scale)
        self.rdb_out = RDB(channels[0], res_scale=res_scale)
        self.rdb_module = nn.ModuleDict({
            f'{scale}_{column}': RDB(channels[scale], res_scale=res_scale)
            for scale in range(3) for column in range(5)
        })
        self.downsample_module = nn.ModuleDict({
            f'{scale}_{column}': DownsampleBlock(channels[scale], channels[scale + 1])
            for scale in range(2) for column in range(3)
        })
        self.upsample_module = nn.ModuleDict({
            f'{scale}_{column}': UpsampleBlock(channels[scale + 1], channels[scale])
            for scale in range(2) for column in range(3, 6)
        })
        self.coefficient = nn.Parameter(torch.ones(3, 6, 2, 64) * 0.5)

    def _blend(self, first, second, scale, column):
        if first.shape[2:] != second.shape[2:]:
            second = F.interpolate(second, size=first.shape[2:],
                                   mode='bilinear', align_corners=False)
        channels = first.shape[1]
        weights = torch.softmax(
            self.coefficient[scale, column, :, :channels], dim=0)
        return (weights[0].view(1, channels, 1, 1) * first +
                weights[1].view(1, channels, 1, 1) * second)

    def forward(self, x):
        features = self.rdb_in(F.relu(self.conv_in(x), inplace=True))
        grid = [[None] * 5 for _ in range(3)]

        grid[0][0] = self.rdb_module['0_0'](features)
        grid[1][0] = self.rdb_module['1_0'](
            self.downsample_module['0_0'](grid[0][0]))
        grid[2][0] = self.rdb_module['2_0'](
            self.downsample_module['1_0'](grid[1][0]))

        for column in range(1, 3):
            grid[0][column] = self.rdb_module[f'0_{column}'](grid[0][column - 1])
            grid[1][column] = self.rdb_module[f'1_{column}'](
                self._blend(grid[1][column - 1],
                            self.downsample_module[f'0_{column}'](grid[0][column]),
                            1, column))
            grid[2][column] = self.rdb_module[f'2_{column}'](
                self._blend(grid[2][column - 1],
                            self.downsample_module[f'1_{column}'](grid[1][column]),
                            2, column))

        grid[2][3] = self.rdb_module['2_3'](grid[2][2])
        grid[1][3] = self.rdb_module['1_3'](
            self._blend(grid[1][2], self.upsample_module['1_3'](grid[2][3]), 1, 3))
        grid[0][3] = self.rdb_module['0_3'](
            self._blend(grid[0][2], self.upsample_module['0_3'](grid[1][3]), 0, 3))

        grid[2][4] = self.rdb_module['2_4'](grid[2][3])
        grid[1][4] = self.rdb_module['1_4'](
            self._blend(grid[1][3], self.upsample_module['1_4'](grid[2][4]), 1, 4))
        grid[0][4] = self.rdb_module['0_4'](
            self._blend(grid[0][3], self.upsample_module['0_4'](grid[1][4]), 0, 4))

        lower = self.upsample_module['1_5'](grid[2][4])
        middle = self._blend(grid[1][4], lower, 1, 5)
        upper = self.upsample_module['0_5'](middle)
        final = self._blend(grid[0][4], upper, 0, 5)
        return torch.clamp(x + self.conv_out(self.rdb_out(final)), 0.0, 1.0)


def resolve_default_weights() -> str | None:
    root = os.path.dirname(os.path.abspath(__file__))
    candidates = [
        os.path.join(root, 'outdoor_haze_best_3_6 (1)'),
        os.path.join(root, 'GridDehazeNet.pth'),
        os.path.join(root, 'grid_dehaze_net.pth'),
        os.path.join(root, '..', 'Models Comparison', 'models',
                     'outdoor_haze_best_3_6 (1)'),
    ]
    return next((path for path in candidates if os.path.exists(path)), None)


def _state_dict(checkpoint):
    if isinstance(checkpoint, nn.Module):
        checkpoint = checkpoint.state_dict()
    if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
        checkpoint = checkpoint['state_dict']
    if not isinstance(checkpoint, dict):
        raise TypeError('checkpoint does not contain a state dictionary')
    if checkpoint and all(key.startswith('module.') for key in checkpoint):
        checkpoint = {key[7:]: value for key, value in checkpoint.items()}
    return checkpoint


def build_griddehazenet(weights_path: str = None, device: str = 'cpu'):
    """Build the model and load a complete compatible checkpoint."""
    resolved_path = weights_path or resolve_default_weights()
    if resolved_path and os.path.exists(resolved_path):
        try:
            model = GridDehazeNet().to(device)
            state = _state_dict(torch.load(resolved_path, map_location=device,
                                            weights_only=False))
            model.load_state_dict(state, strict=True)
            model.eval()
            model.weights_loaded = True
            print(f'[GridDehazeNet] Loaded weights: {resolved_path}')
            return model
        except Exception as exc:
            print(f'[GridDehazeNet] Incompatible weights {resolved_path!r}: {exc}')

    model = GridDehazeNet().to(device)
    model.eval()
    model.weights_loaded = False
    print('[GridDehazeNet] No compatible trained weights found; using untrained model.')
    return model
