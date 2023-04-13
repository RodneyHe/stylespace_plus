import json, torch, cv2
import numpy as np
from torch.utils.data import Dataset
import torchvision.transforms.functional as F

class Transforms():
    def resize(self, image, image_size):
        image = F.resize(image, image_size)
        return image

    def __call__(self, image):
        image = torch.from_numpy(image.transpose((2, 0, 1))).float()
        image.sub_(127.5).div_(128)
        image = self.resize(image, (256, 256))

        return image

class GeneratedDataset(Dataset):
    def __init__(self, args, mode, transform=Transforms()):
        super().__init__()
        self.args = args

        with open(str(args.dataset_path.joinpath("gen_dataset/label.json"))) as f:
            label_data = json.load(f)

        self.image_set = []
        self.z_set = []
        self.transform = transform
        
        dataset_number = len(label_data["images"])
        
        for idx in range(dataset_number):
            if mode == "train" and label_data["images"][str(idx)]["type"] == "train":
                self.image_set.append(label_data["images"][str(idx)]["image_path"])
                self.z_set.append(label_data["images"][str(idx)]["z_path"])

            elif mode == "validate" and label_data["images"][str(idx)]["type"] == "validate":
                self.image_set.append(label_data["images"][str(idx)]["image_path"])
                self.z_set.append(label_data["images"][str(idx)]["z_path"])

    def __len__(self):
        return len(self.image_set)

    def __getitem__(self, idx):
        image = cv2.imread(self.image_set[idx])
        z = torch.from_numpy(np.load(self.z_set[idx])).squeeze()

        if self.transform:
            image = self.transform(image)

        return image, z
