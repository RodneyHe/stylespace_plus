import json
import PIL.Image as Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms.functional as TF
import os

class Transforms():
    def resize(self, image, landmarks, image_size):
        image = TF.resize(image, image_size)
        landmarks = landmarks / 1024 * image_size[0]

        return image, landmarks

    def __call__(self, image, landmarks):
        image, landmarks = self.resize(image, landmarks, (299, 299))
        image = TF.to_tensor(image)

        return image, landmarks

class FaceLandmarksDataset(Dataset):
    def __init__(self, mode, scope="all", transform=None):
        super().__init__()

        with open("./dataset/ffhq_dataset/ffhq-dataset-v2.json") as f:
            data = json.load(f)

        self.image_filenames = []
        self.landmarks = []
        self.transform = transform
        self.root_dir = "dataset/ffhq_dataset"

        for idx in range(len(data)):
            if mode == "training" and data[str(idx)]["category"] == "training":
                self.image_filenames.append(os.path.join(self.root_dir, data[str(idx)]["image"]["file_path"]))
                self.landmarks.append(data[str(idx)]["image"]["face_landmarks"])
            elif mode == "validation" and data[str(idx)]["category"] == "validation":
                self.image_filenames.append(os.path.join(self.root_dir, data[str(idx)]["image"]["file_path"]))
                self.landmarks.append(data[str(idx)]["image"]["face_landmarks"])
        
        if scope != "all" and isinstance(scope, int):
            self.image_filenames = self.image_filenames[:scope]
            self.landmarks = self.landmarks[:scope]

        self.landmarks = torch.tensor(self.landmarks, requires_grad=True)

        assert len(self.image_filenames) == len(self.landmarks)

    def __len__(self):
        return len(self.image_filenames)

    def __getitem__(self, index):
        image = Image.open(self.image_filenames[index])
        landmarks = self.landmarks[index]

        if self.transform:
            image, landmarks = self.transform(image, landmarks)

        return image, landmarks