import torch
from model.network import Network

from writer import Writer
from trainer import Trainer
from general_utils import arglib

def main():
    train_args = arglib.TrainArgs()
    args, str_args = train_args.args, train_args.str_args
    
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    Writer.set_writer(args.results_dir)

    id_model_path = str(args.pretrained_models_path.joinpath("resnet50_scratch_weight.pkl"))
    stylegan_G_path = str(args.pretrained_models_path.joinpath("stylegan2-ffhq-256x256.pkl"))
    landmarks_detector_path = str(args.pretrained_models_path.joinpath('3DDFA/phase1_wpdc_vdc.pth.tar'))

    network = Network(args=args, id_model_path=id_model_path, base_generator_path=stylegan_G_path, 
                      landmarks_detector_path=landmarks_detector_path, device=DEVICE)

    trainer = Trainer(args, network, DEVICE)
    trainer.train()

if __name__ == '__main__':
    main()