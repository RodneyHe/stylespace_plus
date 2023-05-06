import torch
from torch import nn

class ReferenceNetwork(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.model = nn.Sequential(
            nn.Linear(4096, 4096), nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Linear(4096, 4096), nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Linear(4096, 4096), nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Linear(4096, 4928)
        )

    def forward(self, x):
        control_vector = self.model(x)
        return control_vector
    
    def _train(self):
        self.model.train()
        for param in self.model.parameters():
            param.requires_grad = True
    
    def _test(self):
        self.model.eval()
        for param in self.model.parameters():
            param.requires_grad = False
    
    def _save(self, reason):
        torch.save(self.model.state_dict(), str(self.args.weights_dir.joinpath(self.__class__.__name__ + reason + ".pth")))
    
    def _load(self, reason):
        self.model.load_state_dict(torch.load(str(self.args.weights_dir.joinpath(self.__class__.__name__ + reason + ".pth"))))
        self.model.eval()
        print(self.__class__.__name__ + " load checkpoint from: " + str(self.args.weights_dir.joinpath(self.__class__.__name__ + reason + ".pth")))