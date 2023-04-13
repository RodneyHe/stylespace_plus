import os, random, json, argparse, pickle
from tqdm import tqdm
import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms.functional as TF
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import models
from torchmetrics import MultiScaleStructuralSimilarityIndexMeasure
import wandb
import dnnlib, generail_utils.legacy as legacy, utils.general_utils as general_utils

from data_loader.gen_data import FaceLandmarksDataset, Transforms
import model.model as model
from training_modified import networks

train_parser = argparse.ArgumentParser(prog="train", usage="%(prog)s [options]", description="train the style plus mapping function")

# General options
train_parser.add_argument("-s", "--setting_file", type=str, help="use a setting file for config")
train_parser.add_argument("-v", "--version", type=str, default="0.0", help="set the version")
train_parser.add_argument("-m", "--mode", type=str, default="train", help="set the mode to run")

# Dataset options
train_parser.add_argument("-d", "--dataset", type=str, default="gen_dataset", help="set the dataset")

# Training options
train_parser.add_argument("-b", "--batch_size", type=int, default=6, help="set the batch size")
train_parser.add_argument("-e", "--epoch_num", type=int, default=100, help="set the max epoch number")
train_parser.add_argument("-l", "--learning_rate", type=float, default=1e-8, help="set the learning rate")
train_parser.add_argument("-z", "--z_dims", type=int, default=512, help="set the z dimension to run")
train_parser.add_argument("-o", "--out_dir", type=str, default="./output", help="set the output dir")
train_parser.add_argument("-p", "--pretrained_pkl", type=str, default="./pretrained/ffhq.pkl", help="set the mode to run")
train_parser.add_argument("-c", "--cross_frequency", type=int, default=5, help="set the cross_frequency")
train_parser.add_argument("-ph", "--phase", type=str, default="sp", help="set the training phase")

train_args = train_parser.parse_args()

if train_args.setting_file:
    assert train_args.setting_file.endswith(".json")
    with open(train_args.setting_file, "r") as configf:
        config_data = json.load(configf)
        config = config_data
else:
    config = {
        "version": train_args.version,
        "mode": train_args.mode,
        "phase": train_args.phase,
        "dataset": train_args.dataset,
        "batch_size": train_args.batch_size,
        "epoch_num": train_args.epoch_num,
        "lr_rate": train_args.learning_rate,
        "z_dims": train_args.z_dims,
        "out_dir": train_args.out_dir,
        "pretrained_pkl": train_args.pretrained_pkl,
        "cross_frequency": train_args.cross_frequency
    }
run_name = "v" + config["version"] + "_" + config["phase"] +"_" + config["mode"]
run = wandb.init(project="style_plus", name=run_name, config=config, mode="offline")
config = run.config

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# DataLoader
train_dataset = FaceLandmarksDataset("train", transform=Transforms())
train_loader = DataLoader(train_dataset, config.batch_size, shuffle=True)

validate_dataset = FaceLandmarksDataset("validate", transform=Transforms())
validate_loader = DataLoader(validate_dataset, config.batch_size, shuffle=True)

# Load network models
with dnnlib.util.open_url(config.pretrained_pkl) as f:
    pretrained_model = legacy.load_network_pkl(f)
    generator = pretrained_model["G_ema"].to(DEVICE)
    discriminator = pretrained_model["D"].to(DEVICE)

style_generator = networks.Generator(**generator.init_kwargs).to(DEVICE)
style_generator.load_state_dict(generator.state_dict())
style_generator.eval()

style_discriminator = networks.Discriminator(**discriminator.init_kwargs).to(DEVICE)
style_discriminator.load_state_dict(discriminator.state_dict())
style_discriminator.eval()

resnet50 = models.resnet50(weights="DEFAULT").to(DEVICE)
general_utils.load_state_dict(resnet50, "./pretrained/resnet50_scratch_weight.pkl")
resnet50.eval()
resnet_features = model.FeatureExtractor(resnet50, ["fc"])

inception_v3 = models.inception_v3(weights="DEFAULT").to(DEVICE)
inception_features = model.FeatureExtractor(inception_v3, ["fc"])

ref_mapping_network = model.RefMappingNetwork().to(DEVICE)

# if config.phase == "landmark" and config.mode == "training" or config.mode == "validation":
#     if config.mode == "validation":
#         inception_v3_pretrained_dir = f"{config.out_dir}/v{config.version}/saved_model/{config.phase}/inception_v3.pth"
#         ref_mapping_network_pretrained_dir = f"{config.out_dir}/v{config.version}/saved_model/{config.phase}/mapping_network.pth"
#     else:
#         inception_v3_pretrained_dir = f"{config.out_dir}/v{config.version}/saved_model/style_space/inception_v3.pth"
#         ref_mapping_network_pretrained_dir = f"{config.out_dir}/v{config.version}/saved_model/style_space/mapping_network.pth"

#     inception_v3.load_state_dict(torch.load(inception_v3_pretrained_dir))
#     inception_v3.eval()

#     ref_mapping_network.load_state_dict(torch.load(ref_mapping_network_pretrained_dir))
#     ref_mapping_network.eval()

face_lnd_estimator = model.FaceLandmarkEstimator()

# Loss function
l1_loss_fun = nn.L1Loss()
l2_loss_fun = nn.MSELoss()
msssim_loss_fun = MultiScaleStructuralSimilarityIndexMeasure(data_range=1.0).to(DEVICE)

# Optimizer
inception_v3_optimizer = optim.Adam(inception_v3.parameters(), lr=config.lr_rate)
ref_mapping_network_optimizer = optim.Adam(ref_mapping_network.parameters(), lr=config.lr_rate)
# style_space_discriminator_optimizer = optim.RMSprop(style_space_discriminator.parameters(), lr=config.lr_rate)

os.makedirs(config.out_dir, exist_ok=True)
os.makedirs(f"{config.out_dir}/v{config.version}/saved_model/{config.phase}", exist_ok=True)
os.makedirs(f"{config.out_dir}/v{config.version}/saved_img/{config.phase}", exist_ok=True)

# Reference parameters
with open(f"./dataset/{config.dataset}/label.json") as f:
    label_data = json.load(f)

train_set = []
validate_set = []
for idx in range(len(label_data["images"])):
    if label_data["images"][str(idx)]["type"] == "train":
        train_set.append([label_data["images"][str(idx)]["z"][0], label_data["images"][str(idx)]["landmark"]])
    elif label_data["images"][str(idx)]["type"] == "validate":
        validate_set.append([label_data["images"][str(idx)]["z"][0], label_data["images"][str(idx)]["landmark"]])

style_train_mean = torch.tensor(label_data["style_train_mean"]).to(DEVICE)
style_validate_mean = torch.tensor(label_data["style_validate_mean"]).to(DEVICE)

# Training process
for epoch in range(1, config.epoch_num+1):
    if config.mode == "training" or config.mode == "all":
        inception_v3.train()
        ref_mapping_network.train()
        pbar = tqdm(range(1, len(train_loader)+1))
        for step in pbar:
            pbar.set_description(f"epoch {epoch}/training  ")

            ref_images, ref_landmarks, ref_zs, ref_bboxs = next(iter(train_loader))
            ref_images = ref_images.to(DEVICE)
            ref_landmarks = ref_landmarks.to(DEVICE)
            ref_zs = ref_zs.to(DEVICE)
            ref_bboxs = ref_bboxs.to(DEVICE)

            if config.phase == "style_space":
                ref_ws = style_generator.mapping(ref_zs, 0)

                with torch.no_grad():
                    gen_features = resnet_features(ref_images)["fc"][0]

                ref_features = inception_features(TF.resize(ref_images, (299, 299)))["fc"][0]
                spv = ref_mapping_network(torch.cat((gen_features, ref_features), 1))

                gen_images = style_generator.synthesis(ref_ws, spv)
                gen_images_restd = (gen_images * 127.5 + 128).clamp(0, 255) / 255

                sp_train_loss = l1_loss_fun(ref_images, gen_images_restd)
                run.log({"sp_train_loss_0": sp_train_loss.item()})

                # Loss calculation
                inception_v3_optimizer.zero_grad()
                ref_mapping_network_optimizer.zero_grad()

                sp_loss.backward()
                inception_v3_optimizer.step()
                ref_mapping_network_optimizer.step()

                output_dir = f"{config.out_dir}/v{config.version}/saved_img/{config.phase}"
                fname = f"{epoch}_{step}_training.png"
                general_utils.save_images(ref_images=ref_images,
                            modified_gen_images=gen_images_restd, 
                            gen_images=_gen_images_restd, 
                            size=(256, 256), 
                            output_dir=output_dir, 
                            fname=fname)

            elif config.phase == "landmark":
                # gen_ = random.sample(train_set, config.batch_size)
                # gen_zs = [row[0] for row in gen_]
                # gen_lnds = [row[1] for row in gen_]
                # gen_zs = torch.tensor(gen_zs).to(DEVICE)
                # gen_lnds = torch.tensor(gen_lnds).to(DEVICE)

                if step % config.cross_frequency == 0:
                    with torch.no_grad():
                        seed = random.randint(0, 2**23-1)
                        gen_zs = torch.tensor(np.random.RandomState(seed).randn(config.batch_size, generator.z_dim)).to(DEVICE)
                        gen_ws = style_generator.mapping(gen_zs, 0)
                        
                        spv = torch.zeros((config.batch_size, 6048)).to(DEVICE)
                        _gen_images = style_generator.synthesis(gen_ws, spv)
                        _gen_images_destd = (_gen_images * 127.5 + 128).clamp(0, 255)
                        _gen_images_restd = _gen_images_destd / 255

                        _gen_landmarks, unusual_index = face_lnd_estimator(_gen_images_destd)
                        _gen_landmarks = _gen_landmarks / 1024 * 256

                        gen_features = resnet_features(_gen_images_restd)["fc"][0]
                else:
                    with torch.no_grad():
                        gen_ws = style_generator.mapping(ref_zs, 0)
                        gen_features = resnet_features(ref_images)["fc"][0]

                ref_features = inception_features(TF.resize(ref_images, (299, 299)))["fc"][0]
                ref_images = TF.resize(ref_images, (256, 256))

                spv = ref_mapping_network(torch.cat((gen_features, ref_features), 1))

                gen_images = style_generator.synthesis(gen_ws, spv)
                gen_images_destd = (gen_images * 127.5 + 128).clamp(0, 255)
                gen_images_restd = gen_images_destd / 255
                gen_images_restd = TF.resize(gen_images_restd, (256, 256))

                modified_gen_features = resnet_features(gen_images_restd)["fc"][0]

                pred_landmarks, unusual_index = face_lnd_estimator(gen_images_destd)
                pred_landmarks = pred_landmarks.to(DEVICE) / 1024 * 256

                ref_landmarks = ref_landmarks / 1024 * 256

                # Loss calculation
                inception_v3_optimizer.zero_grad()
                ref_mapping_network_optimizer.zero_grad()

                # Feature loss
                feature_loss = l1_loss_fun(modified_gen_features, gen_features)

                # Landmark loss
                # if pred_landmarks.shape[0] != config.batch_size:
                #     assert len(unusual_index) != 0
                #     batch_index = list(range(config.batch_size))

                #     for i in unusual_index:
                #         batch_index.remove(i)

                #     ref_landmarks = torch.index_select(ref_landmarks, 0, torch.tensor(batch_index, device=DEVICE))

                landmark_loss = l2_loss_fun(pred_landmarks[:,17:,:].reshape(config.batch_size, -1),
                                            ref_landmarks[:,17:,:].reshape(config.batch_size, -1))

                # Reconstruction loss
                if step % config.cross_frequency != 0:
                    reconstruction_loss = 0.02 * (0.84 * (1 - msssim_loss_fun(gen_images_restd, ref_images)) \
                                          + 0.16 * l1_loss_fun(gen_images_restd, ref_images))
                else:
                    reconstruction_loss = torch.tensor(0.).to(DEVICE)

                overall_training_loss = feature_loss + 0.001 * landmark_loss + reconstruction_loss

                overall_training_loss.backward()
                inception_v3_optimizer.step()
                ref_mapping_network_optimizer.step()

                if step % config.cross_frequency == 0:
                    output_dir = f"{config.out_dir}/v{config.version}/saved_img/{config.phase}"
                    fname = f"{epoch}_{step}_training_cross.png"
                    general_utils.save_images(ref_images=ref_images, 
                                modified_gen_images=gen_images_restd, 
                                gen_images=_gen_images_restd, 
                                ref_landmarks= ref_landmarks,
                                modified_gen_landmarks= pred_landmarks,
                                gen_landmarks= _gen_landmarks,
                                size=(256, 256), 
                                output_dir=output_dir, 
                                fname=fname)
                elif step % 2 == 0:
                    output_dir = f"{config.out_dir}/v{config.version}/saved_img/{config.phase}"
                    fname = f"{epoch}_{step}_training.png"
                    general_utils.save_images(ref_images=ref_images, 
                                modified_gen_images=gen_images_restd, 
                                gen_images=ref_images, 
                                ref_landmarks=ref_landmarks,
                                modified_gen_landmarks=pred_landmarks,
                                gen_landmarks=ref_landmarks,
                                size=(256, 256), 
                                output_dir=output_dir, 
                                fname=fname)

                run.log({
                    "overall_training_loss": overall_training_loss.item(),
                    "feature_loss": feature_loss.item(),
                    "landmark_loss": landmark_loss.item(),
                    "reconstruction_loss": reconstruction_loss.item()
                })

        # Record training results
        # output_dir = f"{config.out_dir}/v{config.version}/saved_img/{config.phase}"
        # fname = f"{epoch}_{step}_training.png"
        # utils.save_images(ref_images=ref_images, 
        #             modified_gen_images=gen_images_restd, 
        #             gen_images=ref_images, 
        #             size=(256, 256), 
        #             output_dir=output_dir, 
        #             fname=fname)
        # torch.save(ref_mapping_network.state_dict(), f"{config.out_dir}/v{config.version}/saved_model/{config.phase}/mapping_network.pth")
        # torch.save(inception_v3.state_dict(), f"{config.out_dir}/v{config.version}/saved_model/{config.phase}/inception_v3.pth")

    # Validation Process
    if config.mode == "validation" or config.mode == "all":

        inception_v3.eval()
        ref_mapping_network.eval()
        with torch.no_grad():
            pbar = tqdm(range(1, len(validate_loader)+1))
            for step in pbar:
                pbar.set_description(f"epoch {epoch}/validation")

                ref_images, ref_landmarks, ref_zs, ref_bboxs = next(iter(validate_loader))
                ref_images = ref_images.to(DEVICE)
                ref_landmarks = ref_landmarks.to(DEVICE)
                ref_zs = ref_zs.to(DEVICE)
                ref_bboxs = ref_bboxs.to(DEVICE)

                if config.phase == "style_space":
                    ref_ws = style_generator.mapping(ref_zs, 0)

                    with torch.no_grad():
                        gen_features = resnet_features(ref_images)["fc"][0]

                    ref_features = inception_features(ref_images)["fc"][0]
                    spw = ref_mapping_network(torch.cat((gen_features, ref_features), 1))
                    spv = spw * style_validate_mean
                
                    gen_images = style_generator.synthesis(ref_ws, spv)
                    gen_images_restd = (gen_images * 127.5 + 128).clamp(0, 255) / 255

                    sp_loss = l1_loss_fun(ref_images, gen_images)
                    run.log({
                        "sp_validate_loss_0": sp_loss[0].mean().item(),
                        "sp_validate_loss_1": sp_loss[1].mean().item(),
                        "sp_validate_loss_2": sp_loss[2].mean().item(),
                        "sp_validate_loss_3": sp_loss[3].mean().item(),
                        "sp_validate_loss_4": sp_loss[4].mean().item(),
                        "sp_validate_loss_5": sp_loss[5].mean().item()
                    })

        # Recode validation results
        output_dir = f"{config.out_dir}/v{config.version}/saved_img/{config.phase}"
        fname = f"{epoch}_{step}_validation.png"
        general_utils.save_images(ref_images=ref_images, 
                    modified_gen_images=gen_images_restd, 
                    gen_images=_gen_images_restd, 
                    size=(256, 256), 
                    output_dir=output_dir, 
                    fname=fname)
