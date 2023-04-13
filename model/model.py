from torch import nn
import torch
from torch.autograd import Function
import torchvision.transforms.functional as TF
import face_alignment
import face_alignment.utils as fan_utils

class RefMappingNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(4096, 6144), nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Linear(6144, 6144), nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Linear(6144, 6048), nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Linear(6048, 6048)
        )
        self.decoder = nn.Sequential(
            nn.Linear(512, 2048), nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Linear(2048, 4096), nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Linear(4096, 4096), nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Linear(4096, 6048)
        )

    def forward(self, images):
        # Style+ space encoding
        sp_code = self.encoder(images)
        # Style+ decoding
        #spw = self.decoder(sp_code)
        
        return sp_code

class StyleSpaceDiscriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(6048, 3024), nn.LeakyReLU(inplace=True),
            nn.Linear(3024, 1512), nn.LeakyReLU(inplace=True),
            nn.Linear(1512, 756), nn.LeakyReLU(inplace=True),
            nn.Linear(756, 256)
        )

    def forward(self, x):
        x_score = self.network(x)

        return x_score

class FeatureExtractor(nn.Module):
    def __init__(self, model, layers):
        super().__init__()
        self.model = model
        self.layers = layers
        self._features = {layer: torch.empty(0) for layer in layers}

        for layer_id in layers:
            layer = dict([*self.model.named_modules()])[layer_id]
            layer.register_forward_hook(self.save_outputs_hook(layer_id))
    
    def save_outputs_hook(self, layer_id):
        def fn(_, input, output):
            self._features[layer_id] = input
        return fn

    def forward(self, *x):
        _ = self.model(*x)
        return self._features

def transform(point, center, scale, resolution, invert=False):
    """Generate and affine transformation matrix.

    Given a set of points, a center, a scale and a targer resolution, the
    function generates and affine transformation matrix. If invert is ``True``
    it will produce the inverse transformation.

    Arguments:
        point {torch.tensor} -- the input 2D point
        center {torch.tensor or numpy.array} -- the center around which to perform the transformations
        scale {float} -- the scale of the face/object
        resolution {float} -- the output resolution

    Keyword Arguments:
        invert {bool} -- define wherever the function should produce the direct or the
        inverse transformation matrix (default: {False})
    """
    _pt = torch.ones(3)
    _pt[0] = point[0]
    _pt[1] = point[1]

    h = 200.0 * scale
    t = torch.eye(3)
    t[0, 0] = resolution / h
    t[1, 1] = resolution / h
    t[0, 2] = resolution * (-center[0] / h + 0.5)
    t[1, 2] = resolution * (-center[1] / h + 0.5)

    if invert:
        t = torch.inverse(t)

    new_point = (torch.matmul(t, _pt))[0:2]

    return new_point

class DifferentiableCrop(Function):

    @staticmethod
    def forward(ctx, image, center, scale, resolution=256.0):
        ctx.save_for_backward(image, center, scale)
        ctx._input_shape = image.shape
        ctx._input_dtype = image.dtype
        ctx._input_device = image.device

        ul = transform([1, 1], center, scale, resolution, True).int() # Upper-left coordinates
        br = transform([resolution, resolution], center, scale, resolution, True).int() # Bottom-right coordinates

        newDim = [br[1] - ul[1], br[0] - ul[0], image.shape[2]]
        newImg = torch.zeros(newDim, dtype=image.dtype)

        ht = image.shape[0] # Image height
        wd = image.shape[1] # Image width

        newX = [max(1, -ul[0] + 1), min(br[0], wd) - ul[0]]
        newY = [max(1, -ul[1] + 1), min(br[1], ht) - ul[1]]

        oldX = [max(1, ul[0] + 1), min(br[0], wd)]
        oldY = [max(1, ul[1] + 1), min(br[1], ht)]

        newImg[newY[0] - 1:newY[1], newX[0] - 1:newX[1]
               ] = image[oldY[0] - 1:oldY[1], oldX[0] - 1:oldX[1], :]
        newImg = TF.resize(newImg.permute(2,0,1), size=(int(resolution), int(resolution)))
        newImg = newImg.permute(1,2,0).to(image.device)

        return newImg

    @staticmethod
    def backward(ctx, grad_output):
        grad_output = grad_output.permute(2,0,1)

        grad_input = TF.resize(grad_output, (1024, 1024)) # 3 * 1024 * 1024
        grad_input = grad_input.permute(1,2,0) # 1024 * 1024 * 3
        
        return grad_input, None, None, None

class DifferentiableArgmax(Function):

    @staticmethod
    def forward(ctx, input):
        idx = torch.argmax(input, dim=-1)
        ctx._input_shape = input.shape
        ctx._input_dtype = input.dtype
        ctx._input_device = input.device
        ctx.save_for_backward(idx)
        idx += 1

        return idx.to(torch.float32)

    @staticmethod
    def backward(ctx, grad_output):
        idx, = ctx.saved_tensors
        grad_input = torch.zeros(ctx._input_shape, device=ctx._input_device, dtype=ctx._input_dtype)
        grad_input.scatter_(2, idx[..., None], grad_output[..., None])

        return grad_input

class FaceLandmarkEstimator(nn.Module):

    def __init__(self):
        super().__init__()
        self.fa_network = face_alignment.FaceAlignment(face_alignment.LandmarksType._2D, flip_input=False)
        self.differentiableCrop = DifferentiableCrop.apply
        self.differentiableArgmax = DifferentiableArgmax.apply

    def forward(self, images):
        unusual_idx = []
        landmarks = []
        for image_idx, image in enumerate(images):

            d = self.fa_network.face_detector.detect_from_image(image.permute(1,2,0).detach().cpu())

            if len(d) == 1:
                center = torch.tensor([d[0][2] - (d[0][2] - d[0][0]) / 2.0, d[0][3] - (d[0][3] - d[0][1]) / 2.0])
                center[1] = center[1] - (d[0][3] - d[0][1]) * 0.12
                scale = torch.tensor((d[0][2] - d[0][0] + d[0][3] - d[0][1]) / self.fa_network.face_detector.reference_scale)

                #inp = self.differentiableCrop(image.permute(1,2,0), center, scale)

                inp = TF.resize(image, (256, 256))
                inp.div_(255.0).unsqueeze_(0)

                pred_hm = self.fa_network.face_alignment_net(inp)
                B, C, H, W = pred_hm.shape
                pred_hm_reshape = pred_hm.reshape(B, C, H * W)
                
                idx = self.differentiableArgmax(pred_hm_reshape)

                preds = idx.repeat_interleave(2).reshape(B, C, 2)
                preds[:, :, 0] = (preds[:, :, 0] - 1) % W + 1
                preds[:, :, 1] = torch.floor((preds[:, :, 1] - 1) / H) + 1

                for i in range(B):
                    for j in range(C):
                        hm_ = pred_hm[i, j, :]
                        pX, pY = int(preds[i, j, 0]) - 1, int(preds[i, j, 1]) - 1
                        if pX > 0 and pX < 63 and pY > 0 and pY < 63:
                            diff = torch.tensor(
                                [hm_[pY, pX + 1] - hm_[pY, pX - 1],
                                hm_[pY + 1, pX] - hm_[pY - 1, pX]], device=image.device)
                            preds[i, j] += torch.sign(diff) * 0.25

                preds -= 0.5
                preds_orig = torch.zeros_like(preds)
                if center is not None and scale is not None:
                    for i in range(B):
                        for j in range(C):
                            preds_orig[i, j] = transform(preds[i, j], center, scale, H, True)

                pred_lnd = preds_orig
                pred_lnd = pred_lnd.view(68, 2)

                landmarks.append(pred_lnd)
            else:
                landmarks.append(torch.zeros((68, 2), device=image.device))
                unusual_idx.append(image_idx)

        if len(landmarks) != 0:
            for i in range(len(landmarks)):
                landmarks[i] = landmarks[i].unsqueeze(0)
            pred_landmarks = torch.cat(landmarks, dim=0)

            return pred_landmarks, unusual_idx

        return None, None

