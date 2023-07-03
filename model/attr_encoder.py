import torch, torchvision
from torch import nn
import torchvision.transforms.functional as TF
import model.resnet as ResNet
from general_utils import general_utils

class AttrEncoder_Deprecated(nn.Module):
    def __init__(self, args, model_path=None):
        super().__init__()
        self.args = args
        
        self.base_model = ResNet.resnet50(num_classes=8631, include_top=False)
        general_utils.load_state_dict(self.base_model, model_path)
        self.base_model.eval()
        
        self.activation = {}
        self.base_model.avgpool.register_forward_hook(self.get_activation("avgpool"))
        
    def forward(self, x):
        x = self.preprocess(x)
        _ = self.base_model(x)

        return self.activation["avgpool"]
    
    def preprocess(self, img):
        """
        In VGGFace2 The preprocessing is:
            1. Face detection
            2. Expand bbox by factor of 0.3
            3. Resize so shorter side is 256
            4. Crop center 224x224

        In StyleGAN faces are not in-the-wild, we get an image of the head.
        Just cropping a loose center instead of face detection
        """

        # Go from [-1, 1] to [0, 255]
        # img = img * 127.5 + 128

        min_x = int(0.1 * self.args.resolution)
        max_x = int(0.9 * self.args.resolution)
        min_y = int(0.1 * self.args.resolution)
        max_y = int(0.9 * self.args.resolution)

        img = img[:, :, min_x:max_x, min_y:max_y]
        img = TF.resize(img, (256, 256), antialias=True)

        start = (256 - 224) // 2
        img = img[:, :, start: 224 + start, start: 224 + start]

        return img
    
    def get_activation(self, name):
        def hook(model, input, output):
            self.activation[name] = output.squeeze()
        return hook
    
    def _train(self):
        self.base_model.train()
        for param in self.base_model.parameters():
            param.requires_grad = True
    
    def _test(self):
        self.base_model.eval()
        for param in self.base_model.parameters():
            param.requires_grad = False
    
    def _save(self, reason):
        torch.save(self.base_model.state_dict(), str(self.args.weights_dir.joinpath(self.__class__.__name__ + reason + ".pth")))
    
    def _load(self, reason):
        self.base_model.load_state_dict(torch.load(str(self.args.weights_dir.joinpath(self.__class__.__name__ + reason + ".pth"))))
        self.base_model.eval()
        print(self.__class__.__name__ + " loads checkpoint from: " + str(self.args.weights_dir.joinpath(self.__class__.__name__ + reason + ".pth")))

class AttrEncoder(nn.Module):
    def __init__(self, args, attr_model_path=None):
        super().__init__()
        self.args = args
        
        self.base_model = torchvision.models.inception_v3(weights=torchvision.models.Inception_V3_Weights.DEFAULT)
        self.base_model.aux_logits = False
        self.base_model.dropout = nn.Identity()
        self.base_model.fc = nn.Identity()
        self.base_model.eval()
        
        # self.activation = {}
        # self.base_model.avgpool.register_forward_hook(self.get_activation("avgpool"))
        
        if attr_model_path:
            pass
    
    def forward(self, x):
        x = TF.resize(x, (299, 299), antialias=True)
        x = self.base_model(x)
        
        return x
        
    def get_activation(self, name):
        def hook(model, input, output):
            self.activation[name] = output.squeeze()
        return hook
    
    def _train(self):
        self.base_model.train()
        for param in self.base_model.parameters():
            param.requires_grad = True
    
    def _test(self):
        self.base_model.eval()
        for param in self.base_model.parameters():
            param.requires_grad = False
    
    def _save(self, reason):
        torch.save(self.base_model.state_dict(), str(self.args.weights_dir.joinpath(self.__class__.__name__ + reason + ".pth")))
    
    def _load(self, reason):
        self.base_model.load_state_dict(torch.load(str(self.args.weights_dir.joinpath(self.__class__.__name__ + reason + ".pth"))))
        self.base_model.eval()
        print(self.__class__.__name__ + " loads checkpoint from: " + str(self.args.weights_dir.joinpath(self.__class__.__name__ + reason + ".pth")))
