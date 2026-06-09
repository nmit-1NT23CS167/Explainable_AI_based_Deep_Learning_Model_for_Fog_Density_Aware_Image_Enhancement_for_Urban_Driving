"""
AOD-Net: All-in-One Dehazing Network
Paper: "AOD-Net: All-in-One Dehazing Network" (Li et al., 2017)

Architecture: Learns K(x) directly from the hazy image, where
    J(x) = K(x) * I(x) - K(x) + b  (simplified atmospheric scattering model)
"""

import torch
import torch.nn as nn


class AODNet(nn.Module):
    def __init__(self):
        super(AODNet, self).__init__()

        # Feature extraction layers (K-estimation branches)
        self.relu = nn.ReLU(inplace=False)

        self.conv1 = nn.Conv2d(3, 3, 1, 1, 0, bias=True)
        self.conv2 = nn.Conv2d(3, 3, 3, 1, 1, bias=True)
        self.conv3 = nn.Conv2d(6, 3, 5, 1, 2, bias=True)
        self.conv4 = nn.Conv2d(6, 3, 7, 1, 3, bias=True)
        self.conv5 = nn.Conv2d(12, 3, 3, 1, 1, bias=True)

    def forward(self, x):
        source = []
        source.append(x)

        x1 = self.relu(self.conv1(x))
        x2 = self.relu(self.conv2(x1))

        concat1 = torch.cat([x1, x2], dim=1)
        x3 = self.relu(self.conv3(concat1))

        concat2 = torch.cat([x2, x3], dim=1)
        x4 = self.relu(self.conv4(concat2))

        concat3 = torch.cat([x1, x2, x3, x4], dim=1)
        k = self.relu(self.conv5(concat3))

        # Physics-based reconstruction: J = K*I - K + b
        # b is the atmospheric light (assumed ~1.0 for simplicity)
        b = 1.0
        output = k * x - k + b
        output = torch.clamp(output, 0.0, 1.0)
        return output


# Legacy compatibility for checkpoints saved from `model.AODnet`.
AODnet = AODNet
import sys
sys.modules.setdefault("model", sys.modules[__name__])