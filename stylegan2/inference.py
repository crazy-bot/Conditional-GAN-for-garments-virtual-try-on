from PIL import Image
import numpy as np
import dnnlib
import legacy
import torch

device = torch.device('cuda')
## read src image, densepose, pgn mask, 
densePath = '/data/suparna/Data/soccershirt/schalke_new/42_dense.png'
dense = np.array(Image.open(densePath))
dense = dense.transpose(2,0,1)
dense = torch.from_numpy(dense).unsqueeze(0).to(device)

## read cloth image
clothPath = '/data/suparna/Data/soccershirt/BM_cloth.png'
cloth = np.array(Image.open(clothPath))
cloth = cloth.transpose(2,0,1)
cloth = torch.from_numpy(cloth).unsqueeze(0).to(device)

## load the model
network_pkl = 'runs/00005-soccershirt-stylegan2-batch128-noaug-resumecustom/network-snapshot-000512.pkl'
outdir = 'out'
print('Loading networks from "%s"...' % network_pkl)
with dnnlib.util.open_url(network_pkl) as f:
    G = legacy.load_network_pkl(f)['G_ema'].to(device)

    ## get the output
    img,_ = G(dense, cloth, noise_mode='const')
    img = (img.permute(0, 2, 3, 1)).clamp(0, 255).to(torch.uint8)
    Image.fromarray(img[0].cpu().numpy(), 'RGB').save(f'{outdir}/42.png')


