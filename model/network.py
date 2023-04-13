from torch import nn

from model.generator import Generator

class Network(nn.Module):
    def __init__(self, args, id_model_path, base_generator_path,
                 landmark_model_path, device):
        super().__init__()
        self.args = args
        self.generator = Generator(args, id_model_path, base_generator_path, landmark_model_path, device)
    
    def forward(self):
        raise NotImplemented()
    
    def _train(self):
        self.generator._train()
    
    def _test(self):
        self.generator._test()

    def _save(self):
        self.generator._save()