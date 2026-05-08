import importlib
import sys
import torch
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from models.aod_net import AODNet
sys.modules.setdefault('model', importlib.import_module('models.aod_net'))

weights = Path(r'D:\Project\Image enhancement - POC\models\AOD_net_epoch_relu_10.pth')
with torch.serialization.safe_globals([AODNet]):
    ckpt = torch.load(weights, map_location='cpu', weights_only=False)

print(type(ckpt))
if isinstance(ckpt, dict):
    print('keys:', list(ckpt.keys())[:20])
    sd = ckpt['state_dict'] if 'state_dict' in ckpt else ckpt
    print('state_dict sample:', list(sd.keys())[:40])
    print('has conv?', any(k.startswith('conv') for k in sd))
    print('has e_conv?', any(k.startswith('e_conv') for k in sd))
else:
    sd = ckpt.state_dict()
    print('state_dict sample:', list(sd.keys())[:40])
    print('has conv?', any(k.startswith('conv') for k in sd))
    print('has e_conv?', any(k.startswith('e_conv') for k in sd))
