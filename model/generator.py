import torch, dnnlib
from torch import nn
import torchvision.transforms.functional as TF

from general_utils import legacy
from model import landmarks_detector, networks_stylegan2, id_encoder, attr_encoder, reference_autoencoder, reference_network

class Generator(nn.Module):
    def __init__(self, args, id_model_path, base_generator_path,
                 landmarks_detector_path, device):
        super().__init__()
        self.args = args
        
        # Load style generator and freeze its parameters
        with dnnlib.util.open_url(base_generator_path) as f:
            data = legacy.load_network_pkl(f)
            G = data["G_ema"].to(device)
        
        self.stylegan_generator = networks_stylegan2.Generator(**G.init_kwargs).to(device)
        self.stylegan_generator.load_state_dict(G.state_dict())
        self.stylegan_generator.eval()
        
        self.label = torch.zeros([1, G.c_dim], device=device)
        
        for param in self.stylegan_generator.parameters():
            param.requires_grad = False
        
        # Load identity encoder and freeze its parameters
        self.id_encoder = id_encoder.ID_Encoder(args, id_model_path).to(device)
        self.id_encoder._test()
        
        # Inintialize attribute encoder
        self.attr_encoder = attr_encoder.Attr_Encoder(args, id_model_path).to(device)
            
        
        # Load landmarks detector and freeze its parameters
        self.landmarks_detector = landmarks_detector.LandmarksDetector(args, landmarks_detector_path).to(device)
        self.landmarks_detector._test()
        
        # Initialize reference autoencoder
        self.reference_network = reference_network.ReferenceNetwork(args).to(device)
        # self.reference_pose_encoder = reference_autoencoder.ReferenceEncoder(args, 3).to(device)
        # self.reference_pose_decoder = reference_autoencoder.ReferenceDecoder(args, 3).to(device)
    
        # self.reference_expression_encoder = reference_autoencoder.ReferenceEncoder(args, 52).to(device)
        # self.reference_expression_decoder = reference_autoencoder.ReferenceDecoder(args, 52).to(device)
            
    def forward(self, id_img_input, id_z_input, attr_img_input):
        # Identity embedding and attribute embedding
        id_embedding = self.id_encoder(id_img_input)
        attr_embedding = self.attr_encoder(attr_img_input)
        
        # Attribute landmarks
        src_landmarks, src_idx_list = self.landmarks_detector(attr_img_input)

        # Style+ embedding and control vector generation
        feature_tag = torch.concat([id_embedding, attr_embedding], -1)
        
        control_vector = self.reference_network(feature_tag)

        # pose_sp_embedding = self.reference_pose_encoder(feature_tag)
        # pose_control_vector = self.reference_pose_decoder(pose_sp_embedding)
        # expression_sp_embedding = self.reference_expression_encoder(feature_tag)
        # expression_control_vector = self.reference_expression_decoder(expression_sp_embedding)
        # control_vector = pose_control_vector + expression_control_vector

        gen_images = self.stylegan_generator(id_z_input, control_vector)
        
        modified_styles = []
        for block in self.stylegan_generator.synthesis.children():
            for layer in block.children():
                if not isinstance(layer, networks_stylegan2.ToRGBLayer):
                    modified_styles.append(layer.modified_styles.squeeze())
        
        modified_styles = torch.cat(modified_styles, 1)
        
        # Convert RGB to BGR to make the generated image compatible with the landmark detector
        gen_images = gen_images.flip(-3)
        
        return gen_images, id_embedding, attr_embedding, src_landmarks, src_idx_list, modified_styles
    
    def _train(self):
        self.attr_encoder._train()
        self.reference_network._train()
        # self.reference_pose_encoder._train()
        # self.reference_pose_decoder._train()
        # self.reference_expression_encoder._train()
        # self.reference_expression_decoder._train()

    def _test(self):
        self.attr_encoder._test()
        self.reference_network._test()
        # self.reference_pose_encoder._test()
        # self.reference_pose_decoder._test()
        # self.reference_expression_encoder._test()
        # self.reference_expression_decoder._test()
    
    def _save(self, reason):
        self.attr_encoder._save(reason)
        self.reference_network._save(reason)
        # self.reference_pose_encoder._save("_pose"+reason)
        # self.reference_pose_decoder._save("_pose"+reason)
        # self.reference_expression_encoder._save("_expression"+reason)
        # self.reference_expression_decoder._save("_expression"+reason)
    
    def _load(self, reason):
        self.attr_encoder._load(reason)
        self.reference_network._load(reason)
        # self.reference_pose_encoder._load("_pose"+reason)
        # self.reference_pose_decoder._load("_pose"+reason)
        # self.reference_expression_encoder._load("_expression"+reason)
        # self.reference_expression_decoder._load("_expression"+reason)
