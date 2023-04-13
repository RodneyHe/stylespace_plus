import os, sys, random, json, dlib
from tqdm import tqdm
import numpy as np
import torch
import torchvision.transforms.functional as TF
import dnnlib, general_utils.legacy as legacy

from model import stylegan2_ada

def generate_data(base_generator_path, dataset_path, data_number, train_ratio, device):

    # Network & load parameters
    with dnnlib.util.open_url(base_generator_path) as f:
        data = legacy.load_network_pkl(f)
        generator = data["G_ema"].to(device)

    stylegan_generator = stylegan2_ada.Generator(**generator.init_kwargs).to(device)
    stylegan_generator.load_state_dict(generator.state_dict())
    stylegan_generator.eval()
    
    face_detector = dlib.get_frontal_face_detector()
    
    dataset_path = dataset_path + "gen_dataset/"
    os.makedirs(dataset_path, exist_ok=True)
    os.makedirs(dataset_path + "image/", exist_ok=True)

    label = {
        "version": 0.0,
        "train_dataset": "gen",
        "dataset_number": 20000,
        "train_ratio": 0.8,
        "image_size": (1024, 1024),
        "images": {}
    }

    train_num = label["dataset_number"] * label["train_ratio"]
    num = 0
    pbar = tqdm(total=label["dataset_number"], ncols=60)
    while True:
        if num == label["dataset_number"]:
            break
        
        if num % 1000 == 0:
            sub_dir01 = f"image/{1000 * (1 + int(num/1000))}/"
            sub_dir02 = f"z/{1000 * (1 + int(num/1000))}/"
            os.makedirs(dataset_path + sub_dir01, exist_ok=True)
            os.makedirs(dataset_path + sub_dir02, exist_ok=True)

        with torch.no_grad():
            seed = random.randint(0, 2**23-1)
            z = torch.tensor(np.random.RandomState(seed).randn(1, generator.z_dim)).to(device)
            ws = stylegan_generator.mapping(z, 0)
            ctrlv = torch.zeros((1, 6048)).to(device)

            gen_images = stylegan_generator.synthesis(ws, ctrlv)
            gen_images = (gen_images * 127.5 + 128).clamp(0, 255).to(torch.uint8).squeeze()

        rects = face_detector(gen_images.permute(1,2,0).cpu().to(torch.uint8).numpy(), 1)
        
        if rects is not None:
            out_image = TF.to_pil_image(gen_images)
            out_image.save(dataset_path + sub_dir01 + f"{num}.png")
            np.save(dataset_path + sub_dir02 + f"{num}.npy", z.cpu().numpy())

            label["images"][str(num)] = {
                "image_path": dataset_path + sub_dir01 + f"{num}.png",
                "z_path": dataset_path + sub_dir02 + f"{num}.npy"
            }

            if num < train_num:
                label["images"][str(num)]["type"] = "train"
            else:
                label["images"][str(num)]["type"] = "validate"
            
            pbar.update(1)
            num += 1
        else:
            continue

    with open(dataset_path + "label.json", "w") as flabel:
        json.dump(label, flabel)
