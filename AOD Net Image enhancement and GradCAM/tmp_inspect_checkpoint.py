import torch, os
path = r"d:/Project/Image enhancement - POC/models/AOD_net_epoch_relu_10.pth"
print('exists', os.path.exists(path))
if os.path.exists(path):
    try:
        chk = torch.load(path, map_location='cpu', weights_only=False)
        print('type', type(chk))
        if isinstance(chk, dict):
            print('keys', list(chk.keys())[:10])
            if 'state_dict' in chk:
                print('state_dict keys', list(chk['state_dict'].keys())[:20])
    except Exception as e:
        import traceback; traceback.print_exc()
