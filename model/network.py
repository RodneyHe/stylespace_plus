import torch
from torch import nn

from model.generator import Generator

class Network(nn.Module):
    def __init__(self, args, id_model_path, base_generator_path,
                 landmarks_detector_path, device):
        super().__init__()
        self.args = args
        self.generator = Generator(args=args, 
                                   id_model_path=id_model_path, 
                                   base_generator_path=base_generator_path, 
                                   landmarks_detector_path=landmarks_detector_path, 
                                   device=device)

        if args.load_checkpoint:
            self._load(f"_{args.load_checkpoint}")
    
    def forward(self):
        raise NotImplemented()
    
    def _train(self):
        self.generator._train()
    
    def _test(self):
        self.generator._test()

    def _save(self, reason=""):
        self.generator._save(reason)
    
    def _load(self, reason=""):
        self.generator._load(reason)