import json, torch, cv2, os
import numpy as np
from torch.utils.data import Dataset
import torchvision.transforms.functional as TF

class Transforms():
    def resize(self, image, image_size):
        image = TF.resize(image, image_size)
        return image

    def __call__(self, image):
        image = torch.from_numpy(image.transpose((2, 0, 1))).float()
        image.sub_(127.5).div_(128)

        return image.flip(-3) # convert to RGB

class FFHQDataset(Dataset):
    def __init__(self, args, transform=Transforms()):
        super().__init__()
        self.args = args

        with open("./dataset/ffhq256_dataset/label.json") as f:
            label_data = json.load(f)

        self.image_set = []
        self.transform = transform
        dataset_number = len(label_data["images"])

        for key in label_data["images"].keys():
            self.image_set.append(label_data["images"][key]["image_path"])

    def __len__(self):
        return len(self.image_set)

    def __getitem__(self, idx):
        image = cv2.imread(self.image_set[idx])

        if self.transform:
            image = self.transform(image)

        return image