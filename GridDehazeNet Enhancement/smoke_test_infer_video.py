import numpy as np
from models.grid_dehaze_net import build_model
from infer_video import process_frame

if __name__ == '__main__':
    img = np.full((240,320,3), 120, dtype=np.uint8)
    model = build_model(weights_path=None, device='cpu')
    enh, hm, ov = process_frame(img, model, device='cpu', cam=None, run_gradcam=False)
    print('enh type/shape:', type(enh), getattr(enh,'shape',None))
    print('heatmap is None?', hm is None)
    print('overlay is None?', ov is None)
