"""
AOD-Net: All-in-One Dehazing Network (Li et al., 2017)
Layers named conv1..conv5 to match standard pre-trained checkpoints.
"""
import torch
import torch.nn as nn
import os
import importlib
import sys


class AODNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.relu  = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(3,  3,  1, 1, 0, bias=True)
        self.conv2 = nn.Conv2d(3,  3,  3, 1, 1, bias=True)
        self.conv3 = nn.Conv2d(6,  3,  5, 1, 2, bias=True)
        self.conv4 = nn.Conv2d(6,  3,  7, 1, 3, bias=True)
        self.conv5 = nn.Conv2d(12, 3,  3, 1, 1, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1 = self.relu(self.conv1(x))
        x2 = self.relu(self.conv2(x1))
        x3 = self.relu(self.conv3(torch.cat([x1, x2], 1)))
        x4 = self.relu(self.conv4(torch.cat([x2, x3], 1)))
        k  = self.relu(self.conv5(torch.cat([x1, x2, x3, x4], 1)))
        return torch.clamp(k * x - k + 1.0, 0.0, 1.0)

# Legacy alias used in some checkpoints
AODnet = AODNet


def resolve_default_weights() -> str | None:
    root = os.path.dirname(os.path.abspath(__file__))
    candidates = [
        os.path.join(root, 'AOD_net_epoch_relu_10.pth'),
        os.path.join(root, 'AOD_net.pth'),
        os.path.join(root, '..', 'Models Comparison', 'models',
                     'AOD_net_epoch_relu_10.pth'),
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
    if any(key.startswith('e_conv') for key in checkpoint):
        checkpoint = {
            key.replace('e_conv', 'conv', 1): value
            for key, value in checkpoint.items()
        }
    return checkpoint


def build_aodnet(weights_path: str = None,
                 device: str = 'cpu') -> AODNet:
    """Build AOD-Net and load a complete compatible checkpoint."""
    resolved_path = weights_path or resolve_default_weights()
    if resolved_path and os.path.exists(resolved_path):
        try:
            model = AODNet().to(device)
            sys.modules.setdefault('model', importlib.import_module('models.aod_net'))
            checkpoint = torch.load(resolved_path, map_location=device,
                                    weights_only=False)
            model.load_state_dict(_state_dict(checkpoint), strict=True)
            model.eval()
            model.weights_loaded = True
            print(f'[AOD-Net] Loaded weights: {resolved_path}')
            return model
        except Exception as exc:
            print(f'[AOD-Net] Incompatible weights {resolved_path!r}: {exc}')

    model = AODNet().to(device)
    model.eval()
    model.weights_loaded = False
    print('[AOD-Net] No compatible trained weights found; using untrained model.')
    return model
