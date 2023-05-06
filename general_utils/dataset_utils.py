import os, json, dlib
from tqdm import tqdm
import numpy as np
import torch
import torchvision.transforms.functional as TF
import dnnlib, general_utils.legacy as legacy

from model import networks_stylegan2

def generate_data(base_generator_path, dataset_path, dataset_name, data_number, train_ratio, device) -> None:

    # Network & load parameters
    with dnnlib.util.open_url(base_generator_path) as f:
        data = legacy.load_network_pkl(f)
        G = data["G_ema"].to(device)

    stylegan_generator = networks_stylegan2.Generator(**G.init_kwargs).to(device)
    stylegan_generator.load_state_dict(G.state_dict())
    stylegan_generator.eval()
    
    face_detector = dlib.get_frontal_face_detector()
    
    dataset_path = dataset_path + dataset_name
    os.makedirs(dataset_path, exist_ok=True)

    label = {
        "version": 0.0,
        "dataset_name": dataset_name,
        "dataset_number": data_number,
        "train_ratio": train_ratio,
        "image_size": 256,
        "images": {}
    }
    
    c_label = torch.zeros([1, G.c_dim], device=device)
    train_num = label["dataset_number"] * label["train_ratio"]
    num = 0
    pbar = tqdm(total=label["dataset_number"], ncols=80)
    while True:
        if num == label["dataset_number"]:
            pbar.close()
            break

        if num % 1000 == 0:
            sub_dir01 = f"/image/{1000 * (1 + int(num/1000))}/"
            os.makedirs(dataset_path + sub_dir01, exist_ok=True)

        with torch.no_grad():
            z = torch.from_numpy(np.random.RandomState(num).randn(1, G.z_dim)).to(device)
            ctrlv = torch.zeros((1, 7424)).to(device)
            gen_image = stylegan_generator(z, c_label, ctrlv, truncation_psi=1, noise_mode="const")
            
            modified_styles = []
            for block in stylegan_generator.synthesis.children():
                for layer in block.children():
                    if not isinstance(layer, networks_stylegan2.ToRGBLayer):
                        modified_styles = np.append(modified_styles, layer.modified_styles.squeeze().cpu().numpy())

            if num == 0:
                styles = modified_styles[None, ...]
            else:
                styles = np.append(styles, modified_styles[None, ...], 0)

            gen_image = (gen_image * 127.5 + 128).clamp(0, 255).to(torch.uint8).squeeze()

        rects = face_detector(gen_image.permute(1,2,0).cpu().to(torch.uint8).numpy(), 1)
        
        if rects is not None:
            out_image = TF.to_pil_image(gen_image)
            out_image.save(dataset_path + sub_dir01 + f"{num}.png")

            label["images"][str(num)] = {
                "image_path": dataset_path + sub_dir01 + f"{num}.png",
            }

            if num < train_num:
                label["images"][str(num)]["type"] = "train"
            else:
                label["images"][str(num)]["type"] = "validate"
            
            pbar.update(1)
            num += 1
        else:
            continue

    styles_mean = np.mean(styles, 0)
    styles_std = np.std(styles, 0)

    with open(dataset_path + "/label.json", "w") as flabel:
        json.dump(label, flabel)
    
    np.save(dataset_path + "/styles_mean.npy", styles_mean)
    np.save(dataset_path + "/styles_std.npy", styles_std)
