"""
AOD-Net: All-in-One Dehazing Network (Li et al., 2017)
Layers named conv1..conv5 to match standard pre-trained checkpoints.
"""
import torch
import torch.nn as nn


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


def build_aodnet(weights_path: str = None,
                 device: str = 'cpu') -> AODNet:
    """Load AOD-Net, optionally from a .pth checkpoint."""
    import os, importlib, sys
    model = AODNet().to(device)
    if weights_path and os.path.exists(weights_path):
        sys.modules.setdefault('model',
            importlib.import_module('models.aod_net'))
        with torch.serialization.safe_globals([AODNet, AODnet]):
            ckpt = torch.load(weights_path, map_location=device,
                              weights_only=False)
        if isinstance(ckpt, torch.nn.Module):
            state = ckpt.state_dict()
        elif isinstance(ckpt, dict) and 'state_dict' in ckpt:
            state = ckpt['state_dict']
        else:
            state = ckpt
        # remap e_conv → conv if legacy checkpoint
        if any(k.startswith('e_conv') for k in state):
            state = {k.replace('e_conv','conv',1): v
                     for k,v in state.items()}
        model.load_state_dict(state, strict=False)
        print(f'[AOD-Net]  Loaded weights: {weights_path}')
    else:
        print(f'[AOD-Net]  Demo mode (random init)'
              + (f' — {weights_path} not found' if weights_path else ''))
    model.eval()
    return model
