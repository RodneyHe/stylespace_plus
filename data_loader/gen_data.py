import json, torch, cv2, os
import numpy as np
from torch.utils.data import Dataset
import torchvision.transforms.functional as TF

class Transforms():
    def resize(self, image, image_size):
        image = TF.resize(image, image_size, antialias=True)
        return image

    def __call__(self, image):
        image = torch.from_numpy(image.transpose((2, 0, 1))).float()
        image.sub_(127.5).div_(128)

        return image.flip(-3) # convert to RGB

class GeneratedDataset(Dataset):
    def __init__(self, args, mode=None, transform=Transforms()):
        super().__init__()
        self.args = args

        with open(str(args.dataset_path.joinpath("gen_dataset/label.json"))) as f:
            label_data = json.load(f)

        self.image_set = []
        self.transform = transform
        
        dataset_number = len(label_data["images"])
        
        for idx in range(dataset_number):
            self.image_set.append(label_data["images"][str(idx)]["image_path"])
        
        if mode == "train":
            self.image_set = self.image_set[:args.train_data_size]
        elif mode == "validate":
            self.image_set = self.image_set[args.train_data_size:]

    def __len__(self):
        return len(self.image_set)

    def __getitem__(self, idx):
        image = cv2.imread(self.image_set[idx])
        seed = int(os.path.splitext(os.path.basename(self.image_set[idx]))[0])
        z = torch.from_numpy(np.random.RandomState(seed).randn(512))

        if self.transform:
            image = self.transform(image)

        return image, z
