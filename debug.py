import torch, dnnlib
from general_utils import legacy
import torchvision.transforms.functional as TF
import numpy as np
import PIL.Image as Image

from model import networks_stylegan2

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

with dnnlib.util.open_url("./pretrained/stylegan2-ffhq-256x256.pkl") as f:
    data = legacy.load_network_pkl(f)
    G = data["G_ema"].to(device)

seed = 1
stylegan_generator = networks_stylegan2.Generator(**G.init_kwargs).to(device)
stylegan_generator.load_state_dict(G.state_dict())
stylegan_generator.eval()

label = torch.zeros([2, G.c_dim], device=device)
z = torch.from_numpy(np.random.RandomState(seed).randn(2, G.z_dim)).to(device)
ctrlv = torch.zeros([2, 7424], device=device)
img = stylegan_generator(z, label, ctrlv, truncation_psi=1, noise_mode="const")
img = (img * 127.5 + 128).clamp(0, 255).to(torch.uint8)

modified_styles = []
for block in stylegan_generator.synthesis.children():
    for layer in block.children():
        modified_styles.append(layer.modified_styles.squeeze())

modified_styles = torch.cat(modified_styles, 1)

Image.fromarray(img[0].permute(1,2,0).cpu().numpy(), 'RGB').save(f'./test/seed{seed:04d}.png')