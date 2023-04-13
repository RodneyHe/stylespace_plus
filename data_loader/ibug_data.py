import PIL.Image as Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms
import torchvision.transforms.functional as TF
import os, random
from math import cos, sin, radians
import imutils
import numpy as np

# Preprocessing
class Transforms():
    def rotate(self, image, landmarks, angle):
        angle = random.uniform(-angle, angle)

        transformation_matrix = torch.tensor([
            [+cos(radians(angle)), -sin(radians(angle))], 
            [+sin(radians(angle)), +cos(radians(angle))]
        ])

        image = imutils.rotate(np.array(image), angle)

        landmarks = landmarks - 0.5
        new_landmarks = np.matmul(landmarks, transformation_matrix)
        new_landmarks = new_landmarks + 0.5

        return Image.fromarray(image), new_landmarks
    
    def resize(self, image, landmarks, image_size):
        image = TF.resize(image, image_size)

        return image, landmarks
    
    def color_jitter(self, image, landmarks):
        color_jitter = transforms.ColorJitter(
            brightness=0.3, 
            contrast=0.3,
            saturation=0.3, 
            hue=0.1
        )
        image = color_jitter(image)
        return image, landmarks

    def crop_face(self, image, landmarks, crops):
        left = int(crops['left'])
        top = int(crops['top'])
        width = int(crops['width'])
        height = int(crops['height'])

        image = TF.crop(image, top, left, height, width)
        bbox = torch.tensor([left, top, width, height], dtype=torch.float32)

        img_shape = np.array(image).shape
        landmarks = torch.tensor(landmarks) - torch.tensor([[left, top]])
        landmarks = landmarks / torch.tensor([img_shape[1], img_shape[0]])

        return image, landmarks, bbox
    
    def __call__(self, image, landmarks, crops):
        image, landmarks, bbox = self.crop_face(image, landmarks, crops)
        image, landmarks = self.resize(image, landmarks, (256, 256))
        #image, landmarks = self.color_jitter(image, landmarks)
        #image, landmarks = self.rotate(image, landmarks, angle=10)
        
        image = TF.to_tensor(image)
        #image = TF.normalize(image, [0.5], [0.5])
        
        return image, landmarks, bbox

# Data sampler
class FaceLandmarksDataset(Dataset):
    def __init__(self, mode, scope="all", transform=None):

        '''
        mode - choose the training dataset or validate dataset. (option: train, validate)
        scope - choose the scope of dataset. (option: all, afw, helen, ibug, lfpw)
        transform - add the data augmentation.
        '''

        if mode == "train":
            tree = ET.parse("dataset/ibug_300W_large_face_landmark_dataset/labels_ibug_300W_train.xml")
        elif mode == "validate":
            tree = ET.parse("dataset/ibug_300W_large_face_landmark_dataset/labels_ibug_300W_test.xml")
        root = tree.getroot()

        self.image_filenames = []
        self.landmarks = []
        self.crops = []
        self.transform = transform
        self.root_dir = "dataset/ibug_300W_large_face_landmark_dataset"

        if scope == "all":
            for filename in root[2]:
                self.image_filenames.append(os.path.join(self.root_dir, filename.attrib['file']))

                self.crops.append(filename[0].attrib)

                landmark = []
                for num in range(68):
                    x_coordinate = int(filename[0][num].attrib['x'])
                    y_coordinate = int(filename[0][num].attrib['y'])
                    landmark.append([x_coordinate, y_coordinate])
                self.landmarks.append(landmark)
        
            self.landmarks = np.array(self.landmarks).astype('float32')

            assert len(self.image_filenames) == len(self.landmarks)
        else:
            for filename in root[2]:
                if filename.attrib["file"].split("/")[0] == scope:
                    self.image_filenames.append(os.path.join(self.root_dir, filename.attrib['file']))

                    self.crops.append(filename[0].attrib)

                    landmark = []
                    for num in range(68):
                        x_coordinate = int(filename[0][num].attrib['x'])
                        y_coordinate = int(filename[0][num].attrib['y'])
                        landmark.append([x_coordinate, y_coordinate])
                    self.landmarks.append(landmark)
            
            self.landmarks = np.array(self.landmarks).astype('float32')

            assert len(self.image_filenames) == len(self.landmarks)

    def __len__(self):
        return len(self.image_filenames)

    def __getitem__(self, index):
        image = Image.open(self.image_filenames[index]).convert("RGB")
        landmarks = self.landmarks[index]

        if self.transform:
            image, landmarks, bbox = self.transform(image, landmarks, self.crops[index])

        landmarks = landmarks - 0.5

        return image, landmarks, bbox