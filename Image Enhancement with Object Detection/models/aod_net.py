"""
AOD-Net — All-in-One Dehazing Network (Li et al., 2017)
Layer names: conv1..conv5  (matches AOD_net_epoch_relu_10.pth checkpoint)
"""
import torch, torch.nn as nn
import cv2, numpy as np


class AODNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.relu  = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(3,  3,  1, 1, 0, bias=True)
        self.conv2 = nn.Conv2d(3,  3,  3, 1, 1, bias=True)
        self.conv3 = nn.Conv2d(6,  3,  5, 1, 2, bias=True)
        self.conv4 = nn.Conv2d(6,  3,  7, 1, 3, bias=True)
        self.conv5 = nn.Conv2d(12, 3,  3, 1, 1, bias=True)

    def forward(self, x):
        x1 = self.relu(self.conv1(x))
        x2 = self.relu(self.conv2(x1))
        x3 = self.relu(self.conv3(torch.cat([x1, x2], 1)))
        x4 = self.relu(self.conv4(torch.cat([x2, x3], 1)))
        k  = self.relu(self.conv5(torch.cat([x1, x2, x3, x4], 1)))
        return torch.clamp(k * x - k + 1.0, 0.0, 1.0)

# Legacy alias — some checkpoints pickled under this name
AODnet = AODNet


class AODNetEnhancer:
    """
    Convenience wrapper: BGR ndarray → enhanced BGR ndarray.
    Handles loading, device placement, downscale-for-speed, and
    post-processing (contrast stretch + mild sharpening).
    """

    def __init__(self, weights_path: str = None,
                 device: str = "cpu",
                 inference_scale: float = 1.0):
        self.device = torch.device(device)
        self.scale  = inference_scale
        self.model  = AODNet().to(self.device)
        self._load(weights_path)
        self.model.eval()

    def _load(self, path):
        import os, importlib, sys
        if not path or not os.path.exists(path):
            # try any .pth in models/
            models_dir = os.path.join(os.path.dirname(__file__))
            for fn in sorted(os.listdir(models_dir)):
                if fn.endswith(".pth"):
                    path = os.path.join(models_dir, fn); break
        if not path or not os.path.exists(path):
            print("[AOD-Net] No weights found — running untrained (demo only)")
            return
        # Allow legacy pickled model objects
        sys.modules.setdefault("model", importlib.import_module("models.aod_net"))
        with torch.serialization.safe_globals([AODNet, AODnet]):
            ckpt = torch.load(path, map_location=self.device, weights_only=False)
        if isinstance(ckpt, torch.nn.Module):
            state = ckpt.state_dict()
        elif isinstance(ckpt, dict) and "state_dict" in ckpt:
            state = ckpt["state_dict"]
        else:
            state = ckpt
        # Remap e_conv→conv if needed
        if any(k.startswith("e_conv") for k in state):
            state = {k.replace("e_conv", "conv", 1): v for k, v in state.items()}
        self.model.load_state_dict(state)
        print(f"[AOD-Net] Loaded: {path}")

    def enhance(self, frame_bgr: np.ndarray) -> np.ndarray:
        orig_h, orig_w = frame_bgr.shape[:2]
        inp = (cv2.resize(frame_bgr,
                          (max(1, int(orig_w * self.scale)),
                           max(1, int(orig_h * self.scale))))
               if self.scale != 1.0 else frame_bgr)
        t   = self._to_tensor(inp)
        with torch.no_grad():
            out = self.model(t)
        result = self._to_bgr(out)
        if self.scale != 1.0:
            result = cv2.resize(result, (orig_w, orig_h))
        return self._post_process(result)

    def _to_tensor(self, bgr):
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        return (torch.from_numpy(rgb.astype(np.float32) / 255.0)
                .permute(2, 0, 1).unsqueeze(0).to(self.device))

    def _to_bgr(self, t):
        out = t.squeeze(0).cpu().clamp(0, 1)
        return cv2.cvtColor((out * 255).byte().permute(1, 2, 0).numpy(),
                            cv2.COLOR_RGB2BGR)

    @staticmethod
    def _post_process(bgr: np.ndarray) -> np.ndarray:
        """Mild contrast stretch + sharpening — keeps natural colour."""
        # Percentile contrast stretch
        lo, hi = np.percentile(bgr, 2), np.percentile(bgr, 98)
        if hi > lo:
            bgr = np.clip((bgr.astype(np.float32) - lo) / (hi - lo) * 255,
                          0, 255).astype(np.uint8)
        # Unsharp mask (gentle)
        blur    = cv2.GaussianBlur(bgr, (0, 0), sigmaX=2)
        bgr     = cv2.addWeighted(bgr, 1.3, blur, -0.3, 0)
        return bgr
