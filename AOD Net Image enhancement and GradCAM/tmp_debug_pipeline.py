import os, sys
import numpy as np
import torch
import cv2
sys.path.insert(0, os.getcwd())
from models.aod_net import AODNet
from utils.image_utils import preprocess, postprocess, enhance_image, sharpen_image

model = AODNet()
path = 'models/aod_net.pth'
if not os.path.exists(path):
    for fn in os.listdir('models'):
        if fn.lower().endswith('.pth'):
            path = os.path.join('models', fn)
            break
print('using', path)
ckpt = torch.load(path, map_location='cpu')
if isinstance(ckpt, dict) and 'state_dict' in ckpt:
    ckpt = ckpt['state_dict']
model.load_state_dict(ckpt)
img = np.full((100, 100, 3), [120, 150, 200], dtype=np.uint8)
t = preprocess(img, torch.device('cpu'))
out = model(t)
out_np = postprocess(out)
disp = enhance_image(out_np, apply_clahe=True, gamma=1.15)
disp2 = sharpen_image(disp, 0.6)
print('out min,max', out_np.min(), out_np.max(), 'disp min,max', disp.min(), disp.max(), 'disp2 min,max', disp2.min(), disp2.max())
print('out mean', out_np.mean(), 'disp mean', disp.mean(), 'disp2 mean', disp2.mean())
print('out first row', out_np[0, :5, :])
print('disp first row', disp[0, :5, :])
print('disp2 first row', disp2[0, :5, :])
