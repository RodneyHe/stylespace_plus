import os, random
from tqdm import tqdm
import PIL.Image as Image
import numpy as np
import torch
import torchvision.transforms.functional as TF
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import models
import wandb
import dnnlib, generail_utils.legacy as legacy

from data_loader.ffhq_data import FaceLandmarksDataset, Transforms
from archive.model import MappingNetwork, FeatureExtractor, StyleSpaceDiscriminator
from training_modified import networks
import face_alignment

# Train configuration
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

config = {
    "batch_size": 5, 
    "max_epoch": 1000, 
    "lr_rate": 5e-6, 
    "z_dims": 512, 
    "out_dir": "./output", 
    "network_pkl": "./pretrained/ffhq.pkl"
}

run = wandb.init(project="Style_plus", config=config, mode="disabled")
config = run.config

# DataLoader
training_dataset = FaceLandmarksDataset("training", scope=30000, transform=Transforms())
training_loader = DataLoader(training_dataset, config.batch_size, shuffle=True)

validation_dataset = FaceLandmarksDataset("validation", transform=Transforms())
validation_loader = DataLoader(validation_dataset, config.batch_size, shuffle=True)

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

mapping_network = MappingNetwork().to(DEVICE)
style_space_discriminator = StyleSpaceDiscriminator().to(DEVICE)

inception_v3 = models.inception_v3(pretrained=True).to(DEVICE)

inception_features = FeatureExtractor(inception_v3, ["fc"])

fa_network = face_alignment.FaceAlignment(face_alignment.LandmarksType._2D, flip_input=False)

# Loss function & optimizer
mapping_network_optimizer = optim.Adam(mapping_network.parameters(), lr=config.lr_rate)
inception_v3_optimizer = optim.Adam(inception_v3.parameters(), lr=config.lr_rate)
style_space_discriminator_optimizer = optim.Adam(style_space_discriminator.parameters(), lr=config.lr_rate)

# Training process
os.makedirs(config.out_dir, exist_ok=True)

# Style space adversarial training process
for epoch in range(1, 100):
    mapping_network.train()
    inception_v3.train()
    pbar = tqdm(range(1, len(training_loader)+1))
    for step in pbar:
        pbar.set_description(f"epoch {epoch}/training  ")
        

loss_min = np.inf
for epoch in range(1, config.max_epoch+1):

    landmark_training_running_loss = 0
    landmark_validation_running_loss = 0

    mapping_network.train()
    inception_v3.train()
    pbar = tqdm(range(1, len(training_loader)+1))
    for step in pbar:
        pbar.set_description(f"epoch {epoch}/training  ")

        images, landmarks = next(iter(training_loader))
        images = images.to(DEVICE)
        landmarks = landmarks.view(landmarks.shape[0], -1).requires_grad_().to(DEVICE)

        # Style space adversarial learning
        # Reference image encoding
        ref_features = inception_features(images)
        spv = mapping_network(ref_features["fc"][0])

        # StyleGANs image synthesis
        seed = random.randint(0, 2**23-1)
        z = torch.tensor(np.random.RandomState(seed).randn(config.batch_size, generator.z_dim), requires_grad=True).to(DEVICE)
        ws = style_generator.mapping(z, 0)

        generated_images = style_generator.synthesis(ws, spv).detach()
        generated_images = (generated_images * 127.5 + 128).clamp(0, 255).to(torch.uint8)

        original_style = []
        modified_style = []
        for block in style_generator.synthesis.children():
            for layer in block.children():
                if not isinstance(layer, networks.ToRGBLayer):
                    original_style.append(layer.original_style)
                    modified_style.append(layer.modified_style)
        
        original_style = torch.cat(original_style, dim=1)
        modified_style = torch.cat(modified_style, dim=1)

        original_style_score = style_space_discriminator(original_style)
        modified_style_score = style_space_discriminator(modified_style)

        # Train style space discriminator
        style_space_discriminator_optimizer.zero_grad()

        style_space_disc_loss = -original_style_score.mean() + modified_style_score.mean()
        style_space_disc_loss.backward()
        style_space_discriminator_optimizer.step()

        run.log({"style_space_disc_loss": style_space_disc_loss}, commit=False)
        
        # Clip weights of style space discrimator
        for p in style_space_discriminator.parameters():
            p.data.clamp_(-0.01, 0.01)

        # Train mapping network
        if step % 2 == 0:
            inception_v3_optimizer.zero_grad()
            mapping_network_optimizer.zero_grad()

            ref_features = inception_features(images)
            spv = mapping_network(ref_features["fc"][0])
            ws = style_generator.mapping(z, 0)

            generated_images = style_generator.synthesis(ws, spv)
            generated_images = (generated_images * 127.5 + 128).clamp(0, 255).to(torch.uint8)

            original_style = []
            modified_style = []
            for block in style_generator.synthesis.children():
                for layer in block.children():
                    if not isinstance(layer, networks.ToRGBLayer):
                        original_style.append(layer.original_style)
                        modified_style.append(layer.modified_style)

            original_style = torch.cat(original_style, dim=1)
            modified_style = torch.cat(modified_style, dim=1)

            modified_style_score = style_space_discriminator(modified_style)

            modified_style_loss = -modified_style_score.mean()
            modified_style_loss.backward(retain_graph=True)
            inception_v3_optimizer.step()
            mapping_network_optimizer.step()

            run.log({"modified_style_loss": modified_style_loss}, commit=False)

        if step > 200 or epoch > 1:
            # Facial landmark detection
            landmark_batch_loss = 0
            for i in range(config.batch_size):
                pred_ = fa_network.get_landmarks_from_image(generated_images[i].permute(1, 2, 0))

                if pred_ is not None and len(pred_[0]) == 68:
                    pred_landmark = torch.from_numpy(pred_[0]).requires_grad_()
                    landmark_batch_loss += torch.abs(landmarks[i] - pred_landmark.view(-1).to(DEVICE))

            if isinstance(landmark_batch_loss, torch.Tensor):
                inception_v3_optimizer.zero_grad()
                mapping_network_optimizer.zero_grad()

                landmark_training_step_loss = landmark_batch_loss.mean()

                landmark_training_step_loss.backward()
                mapping_network_optimizer.step()
                inception_v3_optimizer.step()

                landmark_training_running_loss = landmark_training_step_loss.item()

        # Critic measurement
        # f_score = style_discriminator(generated_images, 0)
        # r_score = style_discriminator(TF.resize(images, (1024, 1024)), 0)

        # disc_training_loss = -r_score.mean() + f_score.mean()

        # run.log({"disc_training_loss": disc_training_loss.item()})

        run.log({"landmark_training_running_loss": landmark_training_running_loss})

    mapping_network.eval()
    inception_v3.eval()
    with torch.no_grad():
        pbar = tqdm(range(1, len(validation_loader)+1))
        for step in pbar:
            pbar.set_description(f"epoch {epoch}/validation")

            images, landmarks = next(iter(validation_loader))
            images = images.to(DEVICE)
            landmarks = landmarks.view(landmarks.shape[0], -1).to(DEVICE)

            # Reference image encoding
            ref_features = inception_features(images)
            spv = mapping_network(ref_features["fc"][0])

            # StyleGANs image synthesis
            seed = random.randint(0, 2**23-1)
            z = torch.tensor(np.random.RandomState(seed).randn(config.batch_size, generator.z_dim), requires_grad=True).to(DEVICE)
            ws = style_generator.mapping(z, 0)

            # W, Style vector mixing
            generated_images = style_generator.synthesis(ws, spv)
            generated_images = (generated_images * 127.5 + 128).clamp(0, 255).to(torch.uint8)

            # Facial landmark detection
            landmark_batch_loss = 0
            for i in range(config.batch_size):
                pred_ = fa_network.get_landmarks_from_image(generated_images[i].permute(1, 2, 0))
                if pred_ is not None and len(pred_[0]) == 68:
                    pred_landmark = torch.from_numpy(pred_[0])
                    landmark_batch_loss += torch.abs(landmarks[i] - pred_landmark.view(-1).to(DEVICE))

            if isinstance(landmark_batch_loss, torch.Tensor):
                landmark_validation_step_loss = landmark_batch_loss.mean()
                landmark_validation_running_loss += landmark_validation_step_loss

            # Critic measurement
            # f_score = style_discriminator(generated_images, 0)
            # r_score = style_discriminator(TF.resize(images, (1024, 1024)), 0)

            # disc_validate_loss = -r_score.mean() + f_score.mean()

            # run.log({"disc_validate_loss": disc_validate_loss.item()})

            landmark_validation_running_loss = landmark_validation_running_loss / step
            run.log({"landmark_validation_running_loss": landmark_validation_running_loss})

    if landmark_validation_running_loss < loss_min:
        loss_min = landmark_validation_running_loss

        os.makedirs(config.out_dir + f"/saved_model/{run.name}", exist_ok=True)
        os.makedirs(config.out_dir + f"/saved_img/{run.name}/{seed:04d}_{epoch}", exist_ok=True)

        torch.save(mapping_network.state_dict(), config.out_dir + f"/saved_model/{run.name}/mapping_network.pth")
        torch.save(inception_v3.state_dict(), config.out_dir + f"/saved_model/{run.name}/inception_v3.pth")

        ref_img = TF.to_pil_image(images[0])
        gen_img = TF.to_pil_image(TF.resize(generated_images[0], (256, 256)))

        out_img = Image.new("RGB", (512, 256))
        out_img.paste(ref_img, (0, 0))
        out_img.paste(gen_img, (256, 0))
        out_img.save(f"{config.out_dir}/saved_img/{run.name}/{seed:04d}_{epoch}/seed{seed:04d}_gen.png")
