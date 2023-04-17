import torch, numbers
import numpy as np
import pickle
import torchvision.transforms.functional as TF
from PIL import Image, ImageDraw

def load_state_dict(model, fname):
    """
    Set parameters converted from Caffe models authors of VGGFace2 provide.
    See https://www.robots.ox.ac.uk/~vgg/data/vgg_face2/.
    Arguments:
        model: model
        fname: file name of parameters converted from a Caffe model, assuming the file format is Pickle.
    """
    with open(fname, 'rb') as f:
        weights = pickle.load(f, encoding='latin1')

    own_state = model.state_dict()
    for name, param in weights.items():
        if not name.startswith("fc"):
            if name in own_state:
                try:
                    own_state[name].copy_(torch.from_numpy(param))
                except Exception:
                    print(f"{name} is ok", own_state[name].shape, param.size)
                    raise RuntimeError('While copying the parameter named {}, whose dimensions in the model are {} and whose '\
                                       'dimensions in the checkpoint are {}.'.format(name, own_state[name].size(), param.size()))
            else:
                raise KeyError('unexpected key "{}" in state_dict'.format(name))

def gaussian_image(size, sigma, dim=2):
    if isinstance(size, numbers.Number):
        size = [size] * dim
    if isinstance(sigma, numbers.Number):
        sigma = [sigma] * dim

    weight_kernel = 1
    meshgrids = np.meshgrid(*[np.arange(size, dtype=np.float32) for size in size])
    # The gaussian kernel is the product of the
    # gaussian function of each dimension.
    for size, std, mgrid in zip(size, sigma, meshgrids):
        mean = (size - 1) / 2
        weight_kernel *= 1 / (std * np.sqrt(2 * np.pi)) * np.exp(-((mgrid - mean) / std) ** 2 / 2)

    weight_kernel = weight_kernel / np.sum(weight_kernel)
    return weight_kernel

def inverse_gaussian_image(size, sigma, dim=2):
    gauss = gaussian_image(size, sigma, dim)

    # Inversion achieved by max - gauss, but adding min as well to
    # prevent regions of zeros which don't exist in normal gaussian
    inv_gauss = np.max(gauss) + np.min(gauss) - gauss
    inv_gauss = inv_gauss / np.sum(inv_gauss)

    return torch.from_numpy(inv_gauss)

def is_float(tensor):
    """
    Check if input tensor is float32, tensor maybe tf.Tensor or np.array
    """
    return (isinstance(tensor, torch.Tensor) and tensor.dtype != torch.dtypes.uint8) or tensor.dtype != np.uint8

def convert_tensor_to_image(tensor):
    """
    Converts tensor to image, and saturate output's range
    :param tensor: torch.Tensor, dtype float32, range [0,1]
    :return: np.array, dtype uint8, range [0, 255]
    """

    if len(tensor.shape) == 4 and tensor.shape[0] == 1:
        tensor = tensor.squeeze()

    tensor = np.uint8(np.round(tensor))

    return tensor

def concatenate_image(id_image: torch.Tensor, attr_image: torch.Tensor, gen_image: torch.Tensor):
    # Convert BGR to RGB
    id_image = id_image.flip(-3)
    attr_image = attr_image.flip(-3)
    gen_image = gen_image.flip(-3)
    
    # Roughly move to (0, 1)
    id_image = ((id_image + 1) / 2).clamp(0, 1)
    attr_image = ((attr_image + 1) / 2).clamp(0, 1)
    gen_image = ((gen_image + 1) / 2).clamp(0, 1)
    
    # Concatenate images
    con_image = torch.cat((attr_image, gen_image, id_image), 3)
    
    return con_image

def draw_landmarks(image, landmark):
    image_draw = ImageDraw.Draw(image)
    for x, y in zip(landmark[0], landmark[1]):
        image_landmark_coords = [(x-1, y-1), (x+1, y+1)]
        image_draw.ellipse(image_landmark_coords, fill="red")

def save_images(id_images: torch.Tensor, attr_images: torch.Tensor, gen_images: torch.Tensor, attr_landmarks: torch.Tensor, 
                gen_landmarks: torch.Tensor, size: int, output_path: str, sample_number: int = None, landmarks: bool = True):
    
    if sample_number:
        attr_images = attr_images[:sample_number]
        id_images = id_images[:sample_number]
        gen_images = gen_images[:sample_number]
        attr_landmarks = attr_landmarks[:sample_number]
        gen_landmarks = gen_landmarks[:sample_number]
        
    # Convert BGR to RGB
    attr_images = attr_images.flip(-3)
    id_images = id_images.flip(-3)
    gen_images = gen_images.flip(-3)
    
    # Roughly move to (0, 1)
    attr_images = ((attr_images + 1) / 2).clamp(0, 1)
    id_images = ((id_images + 1) / 2).clamp(0, 1)
    gen_images = ((gen_images + 1) / 2).clamp(0, 1)

    out_image = Image.new("RGB", (size*3, size*sample_number))

    attr_coords = [0, 0]
    gen_coords = [size, 0]
    id_coords = [size*2, 0 ]
    for id_image, attr_image, gen_image, attr_landmark, gen_landmark in zip(id_images, attr_images, gen_images, attr_landmarks.tolist(), 
                                                                            gen_landmarks.tolist()):

        attr_image = TF.to_pil_image(TF.resize(attr_image, size))
        id_image = TF.to_pil_image(TF.resize(id_image, size))
        gen_image = TF.to_pil_image(TF.resize(gen_image, size))

        if landmarks:
            draw_landmarks(attr_image, attr_landmark)
            draw_landmarks(gen_image, gen_landmark)

        out_image.paste(attr_image, tuple(attr_coords))
        out_image.paste(gen_image, tuple(gen_coords))
        out_image.paste(id_image, tuple(id_coords))

        attr_coords[1] += size
        gen_coords[1] += size
        id_coords[1] += size

    out_image.save(output_path)
