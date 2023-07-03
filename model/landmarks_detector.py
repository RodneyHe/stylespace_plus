import torch, dlib
import torch.nn as nn
import torchvision.transforms.functional as TF
import torchvision.transforms as transforms
import numpy as np
from model import mobilenet_v1
from general_utils.landmarks_utils import ToTensorGjz, NormalizeGjz, parse_roi_box_from_bbox, predict_68pts, landmarks_calibration
from general_utils.pose_utils import parse_pose

class LandmarksDetector(nn.Module):
    def __init__(self, args, landmarks_detector_path):
        super().__init__()
        self.args = args
        
        arch = "mobilenet_1"
        checkpoint = torch.load(landmarks_detector_path, map_location=lambda storage, loc: storage)["state_dict"]
        self.base_model = getattr(mobilenet_v1, arch)(num_classes=62)  # 62 = 12(pose) + 40(shape) +10(expression)
        model_dict = self.base_model.state_dict()
        
        # because the model is trained by multiple gpus, prefix module should be removed
        for k in checkpoint.keys():
            model_dict[k.replace("module.", "")] = checkpoint[k]
        self.base_model.load_state_dict(model_dict)
        self.base_model.eval()
        
        self.face_detector = dlib.get_frontal_face_detector()
        self.transform = transforms.Compose([ToTensorGjz(), NormalizeGjz(mean=128, std=127.5)])
        
    def forward(self, x, face_detection=True):
        x = self.preprocess(x, face_detection)

        if x is not None:
            x, roi_boxes, idx_sets = x[0], x[1], x[2]
            params = self.base_model(x)
            pts68 = predict_68pts(params, roi_boxes)
            pose, Rs, t3ds = parse_pose(params)
            calib_lnds = landmarks_calibration(pts68, Rs, t3ds)
            return pts68[:, :2, 17:], pose, calib_lnds[:, :2, 17:], idx_sets, pts68
        else:
            return None
    
    # Preprocess
    def preprocess(self, images, face_detection=True):
        if face_detection:
            cropped_images = self.hard_preprocess(images)
        else:
            cropped_images = self.lazy_preprocess(images)

        if cropped_images is not None:
            return cropped_images
        else:
            return None
    
    def lazy_preprocess(self, images):
        images = TF.resize(images, (120, 120), antialias=True)
        return images
        
    def hard_preprocess(self, images):
        idx_list = []
        roi_box_list = []
        cropped_image_list = []
        for idx, image in enumerate(images):
            tmp_image = (image.detach().clone() * 128 + 127.5).clamp(0, 255)
            rects = self.face_detector(tmp_image.permute(1,2,0).to(torch.uint8).cpu().numpy(), 1)

            if len(rects) == 0:
                continue
            
            bbox = [rects[0].left(), rects[0].top(), rects[0].right(), rects[0].bottom()]
            roi_box = parse_roi_box_from_bbox(bbox)
            roi_box = [int(round(_)) for _ in roi_box]
            roi_box_list.append(roi_box)
            idx_list.append(idx)
            
            # Transform input images to tensor and normalize it
            #image = self.transform(image)
            c, h, w = image.shape
            
            sx, sy, ex, ey = roi_box
            dh, dw = ey - sy, ex - sx
            
            if len(image.shape) == 3:
                res = -torch.zeros((3, dh, dw), device=image.device)
                res = res.sub(127.5).div(128)

            if sx < 0:
                sx, dsx = 0, -sx
            else:
                dsx = 0

            if ex > w:
                ex, dex = w, dw - (ex - w)
            else:
                dex = dw

            if sy < 0:
                sy, dsy = 0, -sy
            else:
                dsy = 0

            if ey > h:
                ey, dey = h, dh - (ey - h)
            else:
                dey = dh
            
            res[:, dsy:dey, dsx:dex] = image[:, sy:ey, sx:ex]
            cropped_image = TF.resize(res, (120, 120), antialias=True)
            cropped_image_list.append(cropped_image[None, ...])
        
        if len(idx_list) != 0:
            return torch.cat(cropped_image_list, 0), roi_box_list, set(idx_list)
        else:
            return None
    
    def _train(self):
        self.base_model.train()
        for param in self.base_model.parameters():
            param.requires_grad = True

    def _test(self):
        self.base_model.eval()
        for param in self.base_model.parameters():
            param.requires_grad = False
