import os, random, json
from tqdm import tqdm
import PIL.Image as Image
import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms.functional as TF
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import models
import wandb
import dnnlib, generail_utils.legacy as legacy

from data_loader.gen_data import FaceLandmarksDataset, Transforms
from model.model import RefMappingNetwork, FeatureExtractor, StyleSpaceDiscriminator
from training_modified import networks

# Train configuration
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

config = {
    "version": 0.2,
    "batch_size": 6, 
    "max_epoch": 100, 
    "lr_rate": 5e-6, 
    "z_dims": 512, 
    "out_dir": "./output",
    "network_pkl": "./pretrained/ffhq.pkl"
}

run = wandb.init(project="Style_plus", config=config)
config = run.config

# DataLoader
train_dataset = FaceLandmarksDataset("train", transform=Transforms())
train_loader = DataLoader(train_dataset, config.batch_size, shuffle=True)

# Network
with dnnlib.util.open_url(config.network_pkl) as f:
    data = legacy.load_network_pkl(f)
    generator = data["G_ema"].to(DEVICE)
    discriminator = data["D"].to(DEVICE)

style_generator = networks.Generator(**generator.init_kwargs).to(DEVICE)
style_discriminator = networks.Discriminator(**discriminator.init_kwargs).to(DEVICE)

style_generator.load_state_dict(generator.state_dict())
style_generator.eval()

style_discriminator.load_state_dict(discriminator.state_dict())
style_discriminator.eval()

mapping_network = RefMappingNetwork().to(DEVICE)
# style_space_discriminator = StyleSpaceDiscriminator().to(DEVICE)

# inception_v3 = models.inception_v3().to(DEVICE)
# inception_features = FeatureExtractor(inception_v3, ["fc"])

# fa_network = face_alignment.FaceAlignment(face_alignment.LandmarksType._2D, flip_input=False)

# Loss function & optimizer
sp_loss_fun = nn.L1Loss()

mapping_network_optimizer = optim.RMSprop(mapping_network.parameters(), lr=config.lr_rate)
# inception_v3_optimizer = optim.RMSprop(inception_v3.parameters(), lr=config.lr_rate)
# style_space_discriminator_optimizer = optim.RMSprop(style_space_discriminator.parameters(), lr=config.lr_rate)

mapping_network_scheduler =  optim.lr_scheduler.StepLR(mapping_network_optimizer, step_size=5, gamma=0.1)

# Training process
os.makedirs(config.out_dir, exist_ok=True)

with open("./dataset/gen_dataset/label.json") as f:
    label_data = json.load(f)

train_z_set = []
validate_z_set = []
for idx in range(len(label_data["images"])):
    if label_data["images"][str(idx)]["type"] == "train":
        train_z_set.append(label_data["images"][str(idx)]["z"])
    elif label_data["images"][str(idx)]["type"] == "validate":
        validate_z_set.append(label_data["images"][str(idx)]["z"])

style_mean = torch.tensor(label_data["style_train_mean"]).to(DEVICE)

# Style space adversarial training process
loss_min = np.inf
for epoch in range(1, config.max_epoch+1):

    mapping_network.train()
    # inception_v3.train()
    pbar = tqdm(range(1, len(train_loader)+1))
    for step in pbar:
        pbar.set_description(f"epoch {epoch}")

        ref_images, ref_landmarks, ref_zs, ref_bboxs = next(iter(train_loader))
        ref_images = ref_images.to(DEVICE)
        ref_landmarks = ref_landmarks.to(DEVICE)
        ref_zs = ref_zs.to(DEVICE)
        ref_bboxs = ref_bboxs.to(DEVICE)

        # StyleGANs image synthesis
        ws = style_generator.mapping(ref_zs, 0)

        spw = mapping_network(ref_landmarks.reshape(ref_landmarks.shape[0], -1))
        spv = spw * style_mean

        gen_images = style_generator.synthesis(ws, spv)
        gen_images = (gen_images * 127.5 + 128).clamp(0, 255).to(torch.uint8)

        # original_style = []
        # modified_style = []
        # for block in style_generator.synthesis.children():
        #     for layer in block.children():
        #         if not isinstance(layer, networks.ToRGBLayer):
        #             original_style.append(layer.original_style)
        #             modified_style.append(layer.modified_style)
        
        # original_style = torch.cat(original_style, dim=1)
        # modified_style = torch.cat(modified_style, dim=1)

        # original_style_score = style_space_discriminator(original_style)
        # modified_style_score = style_space_discriminator(modified_style)

        # Train style space discriminator
        # style_space_discriminator_optimizer.zero_grad()

        # style_space_disc_loss = -original_style_score.mean() + modified_style_score.mean()
        # style_space_disc_loss.backward()
        # style_space_discriminator_optimizer.step()

        # run.log({"style_space_disc_loss": style_space_disc_loss}, commit=False)
        
        # Clip weights of style space discrimator
        # for p in style_space_discriminator.parameters():
        #     p.data.clamp_(-0.01, 0.01)

        # Train mapping network
        # inception_v3_optimizer.zero_grad()
        mapping_network_optimizer.zero_grad()

        # ref_features = inception_features(images)
        # spv = mapping_network(ref_features["fc"][0])
        # ws = style_generator.mapping(z, 0)

        # generated_images = style_generator.synthesis(ws, spv)
        # generated_images = (generated_images * 127.5 + 128).clamp(0, 255).to(torch.uint8)

        # original_style = []
        # modified_style = []
        # for block in style_generator.synthesis.children():
        #     for layer in block.children():
        #         if not isinstance(layer, networks.ToRGBLayer):
        #             original_style.append(layer.original_style)
        #             modified_style.append(layer.modified_style)

        # original_style = torch.cat(original_style, dim=1)
        # modified_style = torch.cat(modified_style, dim=1)

        # modified_style_score = style_space_discriminator(modified_style)

        # modified_style_loss = -modified_style_score.mean()
        # modified_style_loss.backward()
        # inception_v3_optimizer.step()
        # mapping_network_optimizer.step()

        # run.log({"modified_style_loss": modified_style_loss.item()})

        ref_spw = torch.zeros_like(spw).to(DEVICE)

        sp_loss = sp_loss_fun(spw, ref_spw)

        run.log({"sp_loss": sp_loss.item()})

        sp_loss.backward()
        mapping_network_optimizer.step()
    
    if sp_loss.item() < 0.1:
        mapping_network_scheduler.step()
    
    os.makedirs(config.out_dir + f"/v{config.version}/saved_model/sp", exist_ok=True)
    os.makedirs(config.out_dir + f"/v{config.version}/saved_img/sp", exist_ok=True)

    torch.save(mapping_network.state_dict(), config.out_dir + f"/v{config.version}/saved_model/sp/mapping_network.pth")
    # torch.save(inception_v3.state_dict(), config.out_dir + f"/saved_model/sp/inception_v3.pth")

    ref_img = TF.to_pil_image(TF.resize(ref_images[0], (256, 256)))
    gen_img = TF.to_pil_image(TF.resize(gen_images[0], (256, 256)))

    out_img = Image.new("RGB", (512, 256))
    out_img.paste(ref_img, (0, 0))
    out_img.paste(gen_img, (256, 0))
    out_img.save(f"{config.out_dir}/v{config.version}/saved_img/sp/{epoch}_gen.png")
