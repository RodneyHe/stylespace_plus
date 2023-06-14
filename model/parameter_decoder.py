import torch
from torch import nn

class ParameterDecoder(nn.Module):
    def __init__(self, args, parameter_dims):
        super().__init__()
        self.args = args
        self.model = nn.Sequential(
            nn.Linear(parameter_dims, 256), nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Linear(256, 512), nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Linear(512, 1024), nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Linear(1024, 2048), nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Linear(2048, 4096), nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Linear(4096, 4928)
        )
        # self.act = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        # self.linear1 = nn.Linear(parameter_dims, 256)
        # self.linear2 = nn.Linear(256, 512)
        # self.linear3 = nn.Linear(512, 1024)
        # self.linear4 = nn.Linear(1024, 2048)
        # self.linear5 = nn.Linear(2048, 4096)
        # self.linear6 = nn.Linear(4096, 4928)

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