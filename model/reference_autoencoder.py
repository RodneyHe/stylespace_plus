import torch
from torch import nn

class ReferenceEncoder(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.encoder = nn.Sequential(
            nn.Linear(4096, 2048), nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Linear(2048, 512), nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Linear(512, 128), nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Linear(128, 3)
        )

    def forward(self, x):
        sp_embedding = self.encoder(x)
        return sp_embedding
    
    def _test(self):
        self.encoder.eval()
        for param in self.encoder.parameters():
            param.requires_grad = False
    
    def _train(self):
        self.encoder.eval()
        for param in self.encoder.parameters():
            param.requires_grad = True
    
    def _save(self):
        torch.save(self.encoder.state_dict(), str(self.args.weights_dir.joinpath(self.__class__.__name__ + ".pth")))

class ReferenceDecoder(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.decoder = nn.Sequential(
            nn.Linear(3, 128), nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Linear(128, 512), nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Linear(512, 2048), nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Linear(2048, 6048)
        )

    def forward(self, sp_embedding):
        control_vector = self.decoder(sp_embedding)
        return control_vector
    
    def _test(self):
        self.decoder.eval()
        for param in self.decoder.parameters():
            param.requires_grad = False
    
    def _train(self):
        self.decoder.eval()
        for param in self.decoder.parameters():
            param.requires_grad = True
    
    def _save(self):
        torch.save(self.decoder.state_dict(), str(self.args.weights_dir.joinpath(self.__class__.__name__ + ".pth")))
