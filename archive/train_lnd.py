import os, random
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

from Data_loader.FFHQ_data import FaceLandmarksDataset, Transforms
import archive.model as model
from training_modified import networks

# Train configuration
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

config = {
    "batch_size": 6, 
    "max_epoch": 100, 
    "lr_rate": 1e-6, 
    "z_dims": 512, 
    "out_dir": "./output", 
    "network_pkl": "./pretrained/ffhq.pkl",
    "sp_pretrained_dir": "./output/saved_model/sp"
}

run = wandb.init(project="Style_plus", config=config, mode="disabled")
config = run.config

# DataLoader
training_dataset = FaceLandmarksDataset("training", scope=30000, transform=Transforms())
training_loader = DataLoader(training_dataset, config.batch_size, shuffle=True)

validation_dataset = FaceLandmarksDataset("validation", transform=Transforms())
validation_loader = DataLoader(validation_dataset, config.batch_size, shuffle=True)

# Network & load parameters
with dnnlib.util.open_url(config.network_pkl) as f:
    data = legacy.load_network_pkl(f)
    generator = data["G_ema"].to(DEVICE)
    discriminator = data["D"].to(DEVICE)

style_generator = networks.Generator(**generator.init_kwargs).to(DEVICE)
style_generator.load_state_dict(generator.state_dict())
style_generator.eval()

style_discriminator = networks.Discriminator(**discriminator.init_kwargs).to(DEVICE)
style_discriminator.load_state_dict(discriminator.state_dict())
style_discriminator.eval()

ref_mapping_network = model.RefMappingNetwork().to(DEVICE)
ref_mapping_network.load_state_dict(torch.load(config.sp_pretrained_dir + "/mapping_network.pth"))
ref_mapping_network.eval()

inception_v3 = models.inception_v3(pretrained=False).to(DEVICE)
inception_v3.load_state_dict(torch.load(config.sp_pretrained_dir + "/inception_v3.pth"))
inception_v3.eval()
inception_features = model.FeatureExtractor(inception_v3, ["fc"])

face_lnd_estimator = model.FaceLandmarkEstimator()

# Loss function & optimizer
landmarks_loss_fun = nn.MSELoss()

inception_v3_optimizer = optim.Adam(inception_v3.parameters(), lr=config.lr_rate)
mapping_network_optimizer = optim.Adam(ref_mapping_network.parameters(), lr=config.lr_rate)
style_discriminator_optimizer = optim.Adam(style_discriminator.parameters(), lr=config.lr_rate)

# Training process
os.makedirs(config.out_dir, exist_ok=True)

for epoch in range(1, config.max_epoch+1):

    gen_disc_training_acc_loss = 0
    gen_training_acc_loss = 0
    landmark_training_acc_loss = 0
    landmark_validation_acc_loss = 0

    inception_v3.train()
    ref_mapping_network.train()
    pbar = tqdm(range(1, len(training_loader)+1))
    for step in pbar:
        pbar.set_description(f"epoch {epoch}/training  ")

        images, landmarks = next(iter(training_loader))
        images = images.to(DEVICE)
        landmarks = landmarks.view(landmarks.shape[0], -1).to(DEVICE)

        # Style space adversarial learning
        # Reference image encoding
        seed = random.randint(0, 2**23-1)
        z = torch.tensor(np.random.RandomState(seed).randn(config.batch_size, generator.z_dim)).to(DEVICE)
        ws = style_generator.mapping(z, 0)

        ref_features = inception_features(images)
        spv = ref_mapping_network(ref_features["fc"][0])

        # StyleGANs image synthesis
        generated_images = style_generator.synthesis(ws, spv)
        generated_images = (generated_images * 127.5 + 128).clamp(0, 255)

        gen_img = TF.to_pil_image(TF.resize(generated_images[0].to(torch.uint8), (256, 256)))
        gen_img.save(f"{config.out_dir}/saved_img/lnd/seed{seed:04d}_{epoch}_{step}_gen.png")

        # # Train stylegan discriminator
        # f_score = style_discriminator(generated_images, 0)
        # r_score = style_discriminator(TF.resize(images, (1024, 1024)), 0)

        # style_discriminator_optimizer.zero_grad()

        # gen_disc_training_loss = -r_score.mean() + f_score.mean()
        # gen_disc_training_loss.backward()
        # style_discriminator_optimizer.step()

        # gen_disc_training_acc_loss += gen_disc_training_loss
        # gen_disc_training_running_loss = gen_disc_training_acc_loss / step

        # run.log({"gen_disc_training_running_loss": gen_disc_training_running_loss}, commit=False)

        # # Clip weights of style space discrimator
        # for p in style_discriminator.parameters():
        #     p.data.clamp_(-0.01, 0.01)

        # # Train mapping network
        # inception_v3_optimizer.zero_grad()
        # mapping_network_optimizer.zero_grad()

        # ref_features = inception_features(images)
        # spv = ref_mapping_network(ref_features["fc"][0])
        # ws = style_generator.mapping(z, 0)

        # generated_images = style_generator.synthesis(ws, spv)
        # generated_images = (generated_images * 127.5 + 128).clamp(0, 255)

        # f_score = style_discriminator(generated_images, 0)
        # gen_training_loss = -f_score.mean()

        # gen_training_loss.backward()
        # mapping_network_optimizer.step()
        # inception_v3_optimizer.step()

        # gen_training_acc_loss += gen_training_loss
        # gen_training_running_loss = gen_training_acc_loss / step

        # run.log({"gen_training_running_loss": gen_training_running_loss})

        # Train facial landmark loss
        inception_v3_optimizer.zero_grad()
        mapping_network_optimizer.zero_grad()

        pred_landmarks, unusual_index = face_lnd_estimator(generated_images)

        if pred_landmarks is not None:
            if pred_landmarks.shape[0] != config.batch_size:
                assert len(unusual_index) != 0

                batch_index = list(range(config.batch_size))

                for i in unusual_index:
                    batch_index.remove(i)
                
                landmarks = torch.index_select(landmarks, 0, torch.tensor(batch_index, device=DEVICE))

            landmarks_loss = landmarks_loss_fun(pred_landmarks.reshape(pred_landmarks.shape[0], -1), landmarks)
            landmarks_training_loss = landmarks_loss.mean()

            landmarks_training_loss.backward()
            mapping_network_optimizer.step()
            inception_v3_optimizer.step()

            landmark_training_acc_loss += landmarks_training_loss
            landmark_training_running_loss = landmark_training_acc_loss / step

            run.log({"landmark_training_running_loss": landmark_training_running_loss})

        if step % 200 == 0:
            ref_img = TF.to_pil_image(images[0])
            gen_img = TF.to_pil_image(TF.resize(generated_images[0], (256, 256)))

            out_img = Image.new("RGB", (512, 256))
            out_img.paste(ref_img, (0, 0))
            out_img.paste(gen_img, (256, 0))
            out_img.save(f"{config.out_dir}/saved_img/lnd/seed{seed:04d}_{epoch}_{step}_gen.png")

    # inception_v3.eval()
    # mapping_network.eval()
    # with torch.no_grad():
    #     pbar = tqdm(range(1, len(validation_loader)+1))
    #     for step in pbar:
    #         pbar.set_description(f"epoch {epoch}/validation")

    #         images, landmarks = next(iter(validation_loader))
    #         images = images.to(DEVICE)
    #         landmarks = landmarks.view(landmarks.shape[0], -1).to(DEVICE)

    #         # Reference image encoding
    #         ref_features = inception_features(images)
    #         spv = mapping_network(ref_features["fc"][0])

    #         # StyleGANs image synthesis
    #         seed = random.randint(0, 2**23-1)
    #         z = torch.tensor(np.random.RandomState(seed).randn(config.batch_size, generator.z_dim), requires_grad=True).to(DEVICE)
    #         ws = style_generator.mapping(z, 0)

    #         # W, Style vector mixing
    #         generated_images = style_generator.synthesis(ws, spv)
    #         generated_images = (generated_images * 127.5 + 128).clamp(0, 255).to(torch.uint8)

    #         # Facial landmark detection
    #         landmark_batch_loss = 0
    #         for i in range(config.batch_size):
    #             pred_ = fa_network.get_landmarks_from_image(generated_images[i].permute(1, 2, 0))
    #             if pred_ is not None and len(pred_[0]) == 68:
    #                 pred_landmark = torch.from_numpy(pred_[0])
    #                 landmark_batch_loss += torch.pow((landmarks[i] - pred_landmark.view(-1).to(DEVICE)), 2)

    #         if isinstance(landmark_batch_loss, torch.Tensor):
    #             landmark_validation_loss = landmark_batch_loss.mean()

    #             f_score = style_discriminator(generated_images, 0)
    #             r_score = style_discriminator(TF.resize(images, (1024, 1024)), 0)

    #             disc_validation_loss = -r_score.mean() + f_score.mean()

    #             run.log({"disc_validation_loss": disc_validation_loss}, commit=False)
    #             run.log({"landmark_validation_loss": landmark_validation_loss})

    if epoch % 2 == 0:
        os.makedirs(config.out_dir + f"/saved_model/lnd", exist_ok=True)
        os.makedirs(config.out_dir + f"/saved_img/lnd", exist_ok=True)

        torch.save(ref_mapping_network.state_dict(), config.out_dir + f"/saved_model/lnd/mapping_network.pth")
        torch.save(inception_v3.state_dict(), config.out_dir + f"/saved_model/lnd/inception_v3.pth")

        ref_img = TF.to_pil_image(images[0])
        gen_img = TF.to_pil_image(TF.resize(generated_images[0], (256, 256)))

        out_img = Image.new("RGB", (512, 256))
        out_img.paste(ref_img, (0, 0))
        out_img.paste(gen_img, (256, 0))
        out_img.save(f"{config.out_dir}/saved_img/lnd/seed{seed:04d}_{epoch}_gen.png")
