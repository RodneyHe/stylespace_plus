import torch
from general_utils.dataset_utils import generate_data

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
generate_data(base_generator_path="./pretrained/stylegan2-ffhq-256x256.pkl", 
              dataset_path="./dataset/", 
              dataset_name="gen_dataset", 
              data_number=20000, 
              train_ratio=0.8, 
              device=DEVICE)