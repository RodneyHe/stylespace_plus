import torch
import numpy as np

from .ddfa import reconstruct_vertex

def parse_roi_box_from_bbox(bbox):
    left, top, right, bottom = bbox
    old_size = (right - left + bottom - top) / 2
    center_x = right - (right - left) / 2.0
    center_y = bottom - (bottom - top) / 2.0 + old_size * 0.14
    size = int(old_size * 1.58)
    roi_box = [0] * 4
    roi_box[0] = center_x - size / 2
    roi_box[1] = center_y - size / 2
    roi_box[2] = roi_box[0] + size
    roi_box[3] = roi_box[1] + size
    return roi_box

def _predict_vertices(params, roi_bboxes, dense, transform=True):
    vertex_list = []
    for param, roi_bbox in zip(params, roi_bboxes):
        vertex = reconstruct_vertex(param, dense=dense)
        sx, sy, ex, ey = roi_bbox
        scale_x = (ex - sx) / 120
        scale_y = (ey - sy) / 120
        vertex[0, :] = vertex[0, :] * scale_x + sx
        vertex[1, :] = vertex[1, :] * scale_y + sy

        s = (scale_x + scale_y) / 2
        vertex[2, :] *= s
        vertex_list.append(vertex[None, ...])

    return torch.cat(vertex_list, 0)

def predict_68pts(param, roi_box):
    return _predict_vertices(param, roi_box, dense=False)

def landmarks_calibration(landmarks: torch.Tensor, Rs: torch.Tensor, t3ds: torch.Tensor):
    '''
    Args:
        landmarks: (N, 3, 68). facial landmarks coordinates
        R: (3,3). rotation matrix
    Returns:
        calibrated_landmarks: (N, 68, 3)
    '''
    assert len(landmarks.shape) == 3 
    assert len(Rs.shape) == 3
    assert landmarks.shape[0] == Rs.shape[0]
    
    #batch_size = landmarks.shape[0]
    
    #homo_lnds = torch.cat([landmarks.permute(0, 2, 1), torch.ones((batch_size, 68, 1), device=landmarks.device)], -1)
    
    # calib_lnds_list = []
    # for homo_lnd, R, t3d in zip(homo_lnds, Rs, t3ds):
    #     # Construct translation matrix
        
    #     T = torch.eye(4, device=landmarks.device)
    #     T[0, 3] = -t3d[0]
    #     T[1, 3] = -t3d[1]
    #     T[2, 3] = -t3d[2]
        
    #     untrans_lnds = homo_lnd.matmul(T.T)
        
    #     calib_lnds = untrans_lnds[:, :3].matmul(R.T.inverse())
    #     calib_lnds_list.append(calib_lnds[None,...])
    
    calib_lnds_list = []
    for homo_lnd, R, t3d in zip(landmarks.permute(0,2,1), Rs, t3ds):
        # Construct translation matrix
        
        # T = torch.eye(4, device=landmarks.device)
        # T[0, 3] = -t3d[0]
        # T[1, 3] = -t3d[1]
        # T[2, 3] = -t3d[2]
        
        # untrans_lnds = homo_lnd.matmul(T.T)
        
        calib_lnds = homo_lnd[:, :3].matmul(R.T.inverse())
        calib_lnds_list.append(calib_lnds[None,...])

    return torch.cat(calib_lnds_list, 0).permute(0, 2, 1)

class ToTensorGjz(object):
    def __call__(self, pic):
        if isinstance(pic, np.ndarray):
            img = torch.from_numpy(pic.transpose((2, 0, 1)))
            return img.float()
        elif isinstance(pic, torch.Tensor):
            return pic

    def __repr__(self):
        return self.__class__.__name__ + '()'

class NormalizeGjz(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        tensor.sub_(self.mean).div_(self.std)
        return tensor