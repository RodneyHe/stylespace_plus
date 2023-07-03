import torch, os
import numpy as np
import torchvision.transforms.functional as TF
from pathlib import Path

from torch.utils.data.dataloader import DataLoader
from data_loader.ffhq_data import FFHQDataset

from model.generator import Generator

import cv2 as cv

def generate_image(generator, ws, attr_image, style_padding):

    with torch.no_grad():
        attr_embed = generator.attr_encoder(attr_image)
        ctrl_vec = generator.reference_network(attr_embed)
        #ctrl_vec = torch.cat([style_padding[0], ctrl_vec, style_padding[1]], -1)
        #ctrl_vec = torch.cat([ctrl_vec, style_padding], -1)

        gen_image = generator.stylegan_generator.synthesis(ws, ctrl_vec)
        gen_image = ((gen_image + 1) / 2).clamp(0, 1)
        
        return gen_image

def evaluate(args, configs: dict, device):
    id_model_path = configs["id_model_path"]
    stylegan_G_path = configs["stylegan_G_path"]
    landmarks_model_path = configs["landmarks_model_path"]
    batch_size = configs["batch_size"]

    results_dir = args.results_dir
    weights_dir = results_dir.joinpath(args.name+"/weights")

    generator = Generator(args=args, id_model_path=id_model_path, 
                          base_generator_path=stylegan_G_path, 
                          landmarks_detector_path=landmarks_model_path, 
                          device=device)
    
    generator._load("")

    # 23011 23866

    ffhq_dataset = FFHQDataset(args)
    ffhq_dataloader = DataLoader(ffhq_dataset, batch_size=batch_size, shuffle=True, pin_memory=True)
    ffhq_iter = iter(ffhq_dataloader)

    attr_path_list = ["./dataset/ffhq256_dataset/image/07000/07983.png",
                  "./dataset/ffhq256_dataset/image/04000/04986.png",
                  "./dataset/ffhq256_dataset/image/07000/07024.png"]

    attr_list = []
    for attr_path in attr_path_list:
        attr_image = cv.imread(attr_path)

        attr_image = torch.from_numpy(attr_image.transpose((2, 0, 1))).float()
        attr_image.sub_(127.5).div_(128)

        attr_image = attr_image.flip(-3)  # convert to RGB
        attr_list.append(attr_image[None, ...])

    attr_images = torch.cat(attr_list, 0).to(device)

    id_path_list = ["./dataset/gen_dataset/image/14000/13006.png",
                    "dataset/gen_dataset/image/43000/42024.png",
                    "dataset/gen_dataset/image/46000/45009.png"]

    id_image_list = []
    id_seed_list = []
    for id_path in id_path_list:
        id_image = cv.imread(id_path)

        id_image = torch.from_numpy(id_image.transpose((2, 0, 1))).float()
        id_image.sub_(127.5).div_(128)

        id_image = id_image.flip(-3)  # convert to RGB
        id_image_list.append(id_image[None, ...])

        id_image_seed = int(os.path.splitext(os.path.basename(id_path))[0])
        id_seed_list.append(id_image_seed)

    id_images = torch.cat(id_image_list, 0).to(device)

    style_padding = torch.zeros((1, 2368)).to(device)
    style_padding1 = torch.zeros((1, 1536)).to(device)
    style_padding2 = torch.zeros((1, 576)).to(device)

    # Sample one batch and editing
    attr_images_from_real = next(ffhq_iter)
    attr_images_from_real = attr_images_from_real.to(device)

    # Generate identiry images
    # z = torch.randn((batch_size, 512)).to(device)
    # ws = generator.stylegan_generator.mapping(z, 0)

    zs_list = []
    for id_seed in id_seed_list:
        z = torch.from_numpy(np.random.RandomState(id_seed).randn(1, 512)).to(device)
        zs_list.append(z)

    zs = torch.cat(zs_list, 0)
    ws = generator.stylegan_generator.mapping(zs, 0)
    
    with torch.no_grad():
        ctrl_vec = torch.zeros((batch_size, 4928)).to(device)
        gen_id_images = generator.stylegan_generator.synthesis(ws, ctrl_vec)
        gen_id_images = ((gen_id_images + 1) / 2).clamp(0, 1)

    gen_images_list = []
    for attr_image in attr_images:
        gen_images = []
        for w in ws:
            gen_image = generate_image(generator, w[None,...], attr_image[None,...], style_padding)
            gen_images.append(gen_image.squeeze())
        gen_images_list.append(torch.cat(gen_images, 1))
    
    gen_images_tensor = torch.cat(gen_images_list, -1)
    
    attr_images = ((attr_images + 1) / 2).clamp(0, 1)
        
    gen_id_images = torch.cat([gen_id_image for gen_id_image in gen_id_images], 1)
    attr_images = torch.cat([attr_images for attr_images in attr_images], 2)

    gen_results = torch.cat([gen_id_images, gen_images_tensor], 2)
    attr_images = torch.cat([torch.ones(3, 256, 256).to(device), attr_images], 2)
    gen_results = torch.cat([attr_images, gen_results], 1)
    
    return gen_results


