import os
import sys
import torch
import importlib
import numpy as np

sys.path.insert(0, os.getcwd())
from models.aod_net import AODNet
import models.aod_net as model_mod
from utils.image_utils import preprocess, postprocess, enhance_image, sharpen_image

safe_globals = [AODNet]
if hasattr(model_mod, "AODnet"):
    safe_globals.append(model_mod.AODnet)

with torch.serialization.safe_globals(safe_globals):
    checkpoint = torch.load("models/AOD_net_epoch_relu_10.pth", map_location="cpu", weights_only=False)

if isinstance(checkpoint, torch.nn.Module):
    state = checkpoint.state_dict()
elif isinstance(checkpoint, dict) and "state_dict" in checkpoint:
    state = checkpoint["state_dict"]
else:
    state = checkpoint

state_keys = list(state.keys()) if isinstance(state, dict) else None
print("[OK] Checkpoint loaded")

model = AODNet()
model.load_state_dict(state)
model.eval()

img = np.zeros((128, 128, 3), dtype=np.uint8)
img[..., 0] = 60
img[..., 1] = 120
img[..., 2] = 200
print("Input image (BGR) blue-rich, hazy-like: mean", img.mean())

t = preprocess(img, torch.device("cpu"))
out = model(t)
enhanced_np = postprocess(out)
print("Raw model output stats: min={:.0f}, max={:.0f}, mean={:.1f}".format(enhanced_np.min(), enhanced_np.max(), enhanced_np.mean()))
print("Raw output looks good?", enhanced_np.min() > 50 and enhanced_np.max() < 250)

# Test minimal post-processing
minimal = enhanced_np.copy()
minimal = sharpen_image(minimal, strength=0.2)
print("After minimal sharpening: min={:.0f}, max={:.0f}, mean={:.1f}".format(minimal.min(), minimal.max(), minimal.mean()))

# Test old aggressive processing
aggressive = enhance_image(enhanced_np, apply_clahe=True, gamma=0.98, saturation_scale=1.1)
aggressive = sharpen_image(aggressive, strength=0.45)
print("After aggressive enhance: min={:.0f}, max={:.0f}, mean={:.1f}".format(aggressive.min(), aggressive.max(), aggressive.mean()))

# Test new gentle processing
gentle = enhance_image(enhanced_np, apply_clahe=False, gamma=1.0, saturation_scale=1.0)
gentle = sharpen_image(gentle, strength=0.3)
print("After gentle enhance:     min={:.0f}, max={:.0f}, mean={:.1f}".format(gentle.min(), gentle.max(), gentle.mean()))

print("\n[RESULT] Gentle processing: contrast stretching only, no CLAHE/gamma/saturation boost")
