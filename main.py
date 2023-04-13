import os
import torch

import sys
import logging
from model.network import Network
from model.generator import Generator

from writer import Writer
from trainer import Trainer
from general_utils import arglib

def init_logger(args):
    root_logger = logging.getLogger()

    level = logging.DEBUG if args.log_debug else logging.INFO
    root_logger.setLevel(level)

    file_handler = logging.FileHandler(f'{args.results_dir}/log.txt')
    console_handler = logging.StreamHandler()

    datefmt = '%Y-%m-%d %H:%M:%S'
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s', datefmt)

    file_handler.setLevel(level)
    console_handler.setLevel(level)

    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)

    root_logger.addHandler(file_handler)
    root_logger.addHandler(console_handler)

    pil_logger = logging.getLogger('PIL.PngImagePlugin')
    pil_logger.setLevel(logging.INFO)

def main():
    train_args = arglib.TrainArgs()
    args, str_args = train_args.args, train_args.str_args
    
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    Writer.set_writer(args.results_dir)

    id_model_path = str(args.pretrained_models_path.joinpath("resnet50_scratch_weight.pkl"))
    stylegan_G_path = str(args.pretrained_models_path.joinpath("ffhq.pkl"))
    landmarks_model_path = str(args.pretrained_models_path.joinpath('3DDFA/phase1_wpdc_vdc.pth.tar'))

    network = Network(args=args, id_model_path=id_model_path, base_generator_path=stylegan_G_path, 
                      landmark_model_path=landmarks_model_path, device=DEVICE)

    regulizing_network = Trainer(args, network, DEVICE)
    regulizing_network.train()

if __name__ == '__main__':
    main()