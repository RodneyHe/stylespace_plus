import torch, dnnlib
from torch import nn
import torchvision.transforms.functional as F

from general_utils import legacy
from model import landmarks_detector, stylegan2_ada, id_encoder, attr_encoder, reference_autoencoder

class Generator(nn.Module):
    def __init__(self, args, id_model_path, base_generator_path,
                 landmarks_detector_path, device, attr_model_path=None):
        super().__init__()
        self.args = args
        
        # Load style generator and freeze its parameters
        with dnnlib.util.open_url(base_generator_path) as f:
            data = legacy.load_network_pkl(f)
            G = data["G_ema"].to(device)
        
        self.stylegan_generator = stylegan2_ada.Generator(**G.init_kwargs).to(device)
        self.stylegan_generator.load_state_dict(G.state_dict())
        self.stylegan_generator.eval()
        
        for param in self.stylegan_generator.parameters():
            param.requires_grad = False
        
        # Load identity encoder and freeze its parameters
        self.id_encoder = id_encoder.ID_Encoder(args, id_model_path).to(device)
        self.id_encoder._test()
        
        # Load attribute encoder
        self.attr_encoder = attr_encoder.Attr_Encoder(args, id_model_path).to(device)
        
        # Load landmarks detector and freeze its parameters
        self.landmarks_detector = landmarks_detector.LandmarksDetector(args, landmarks_detector_path).to(device)
        self.landmarks_detector._test()
        
        # Load reference autoencoder
        self.reference_encoder = reference_autoencoder.ReferenceEncoder(args).to(device)
        self.reference_decoder = reference_autoencoder.ReferenceDecoder(args).to(device)
    
    def forward(self, id_img_input, id_z_input, attr_img_input):
        # Identity embedding and attribute embedding
        id_embedding = self.id_encoder(id_img_input)
        attr_embedding = self.attr_encoder(attr_img_input)
        
        # Attribute landmarks
        attr_landmarks, attr_idx_list = self.landmarks_detector(attr_img_input)

        # Style+ embedding and control vector generation
        feature_tag = torch.concat([id_embedding, attr_embedding], -1)
        sp_embedding = self.reference_encoder(feature_tag)
        control_vector = self.reference_decoder(sp_embedding)
        
        gen_images = self.stylegan_generator(id_z_input, control_vector)
        
        gen_images = F.resize(gen_images, (256, 256))
        
        # Convert RGB to BGR to make the generated image compatible with the landmark detector
        gen_images = gen_images.flip(-3)
        
        return gen_images, id_embedding, attr_embedding, attr_landmarks, attr_idx_list
    
    def _train(self):
        self.attr_encoder._train()
        self.reference_encoder._train()
        self.reference_decoder._train()

    def _test(self):
        self.attr_encoder._test()
        self.reference_encoder._test()
        self.reference_decoder._test()
    
    def _save(self):
        self.attr_encoder._save()
        self.reference_encoder._save()
        self.reference_decoder._save()
