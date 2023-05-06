import torch
from torch import nn

class ReferenceEncoder(nn.Module):
    def __init__(self, args, embedding_dim):
        super().__init__()
        self.args = args
        self.encoder = nn.Sequential(
            nn.Linear(4096, 2048), nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Linear(2048, 512), nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Linear(512, 128), nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Linear(128, embedding_dim)
        )

    def forward(self, x):
        sp_embedding = self.encoder(x)
        return sp_embedding

    def _train(self):
        self.encoder.train()
        for param in self.encoder.parameters():
            param.requires_grad = True
    
    def _test(self):
        self.encoder.eval()
        for param in self.encoder.parameters():
            param.requires_grad = False
    
    def _save(self, reason):
        torch.save(self.encoder.state_dict(), str(self.args.weights_dir.joinpath(self.__class__.__name__ + reason + ".pth")))
        
    def _load(self, reason):
        self.encoder.load_state_dict(torch.load(str(self.args.weights_dir.joinpath(self.__class__.__name__ + reason + ".pth"))))
        self.encoder.eval()
        print(self.__class__.__name__ + " load checkpoint from: " + str(self.args.weights_dir.joinpath(self.__class__.__name__ + reason + ".pth")))

class ReferenceDecoder(nn.Module):
    def __init__(self, args, embedding_dim):
        super().__init__()
        self.args = args
        self.decoder = nn.Sequential(
            nn.Linear(embedding_dim, 128), nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Linear(128, 512), nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Linear(512, 2048), nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Linear(2048, 4928)
        )

    def forward(self, sp_embedding):
        control_vector = self.decoder(sp_embedding)
        return control_vector
    
    def _train(self):
        self.decoder.train()
        for param in self.decoder.parameters():
            param.requires_grad = True
    
    def _test(self):
        self.decoder.eval()
        for param in self.decoder.parameters():
            param.requires_grad = False
    
    def _save(self, reason):
        torch.save(self.decoder.state_dict(), str(self.args.weights_dir.joinpath(self.__class__.__name__ + reason + ".pth")))
    
    def _load(self, reason):
        self.decoder.load_state_dict(torch.load(str(self.args.weights_dir.joinpath(self.__class__.__name__ + reason + ".pth"))))
        self.decoder.eval()
        print(self.__class__.__name__ + " load checkpoint from: " + str(self.args.weights_dir.joinpath(self.__class__.__name__ + reason + ".pth")))
