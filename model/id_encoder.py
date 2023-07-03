import torch, pickle
from torch import nn
import torchvision.transforms.functional as TF 
import model.inception_resnet_v1 as inception_resnet_v1
from general_utils import general_utils

class ID_Encoder(nn.Module):
    def __init__(self, args, model_path):
        super().__init__()
        self.args = args
        self.base_model = inception_resnet_v1.InceptionResnetV1(pretrained="vggface2", pretrained_model_path=model_path)
        self.base_model.eval()

    def forward(self, x):
        x = self.preprocess(x)
        x = self.base_model(x)

        return x
    
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
    
    def _train(self):
        self.base_model.train()
        for param in self.base_model.parameters():
            param.requires_grad = True
    
    def _test(self):
        self.base_model.eval()
        for param in self.base_model.parameters():
            param.requires_grad = False