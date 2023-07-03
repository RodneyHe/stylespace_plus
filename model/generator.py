import torch, dnnlib
from torch import nn
import torchvision.transforms.functional as TF
import numpy as np

from general_utils import legacy
from model import landmarks_detector, networks_stylegan2, id_encoder, attr_encoder, reference_autoencoder, reference_network, parameter_decoder

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
        self.attr_encoder = attr_encoder.AttrEncoder(args).to(device)
        
        # Load landmarks detector and freeze its parameters
        self.landmarks_detector = landmarks_detector.LandmarksDetector(args, landmarks_detector_path).to(device)
        self.landmarks_detector._test()
        
        # Initialize reference autoencoder
        # 
        # self.reference_pose_encoder = reference_autoencoder.ReferenceEncoder(args, 3).to(device)
        # self.reference_pose_decoder = reference_autoencoder.ReferenceDecoder(args, 3).to(device)
    
        # self.reference_expression_encoder = reference_autoencoder.ReferenceEncoder(args, 52).to(device)
        # self.reference_expression_decoder = reference_autoencoder.ReferenceDecoder(args, 52).to(device)

        # Initialize orthogonal nn.Parameter
        if args.parameter_embedding:
            #self.pose_basis = nn.init.orthogonal_(nn.Parameter(torch.randn(3, 3))).to(device)
            
            # finetune attrencoder
            self.attr_encoder.base_model.load_state_dict(torch.load("./output/exp_18-conv0--2/weights/AttrEncoder.pth"))
            self.attr_encoder.eval()
            self.attr_encoder._test()
            
            # frozen pose encoder
            self.pose_parameter_decoder = parameter_decoder.ParameterDecoder(args, parameter_type="pose").to(device)
            self.pose_parameter_decoder.model.load_state_dict(torch.load("./output/exp_18-conv0--2/weights/ReferenceNetwork.pth"))
            self.pose_parameter_decoder.eval()
            self.pose_parameter_decoder._test()

            # train expression encoder
            self.expression_parameter_decoder = parameter_decoder.ParameterDecoder(args, parameter_type="expression").to(device)
            self.style_padding = torch.zeros((self.args.batch_size, 2880), requires_grad=False).to(device)
        else:
            self.reference_network = reference_network.ReferenceNetwork(args).to(device)
            self.style_padding1 = torch.zeros((self.args.batch_size, 1536), requires_grad=False).to(device)
            self.style_padding2 = torch.zeros((self.args.batch_size, 576), requires_grad=False).to(device)
            
    def forward(self, id_imgs, id_zs, attr_imgs):

        # Attribute landmarks
        src_lnd_results = self.landmarks_detector(attr_imgs.flip(-3)) # the input is converted to BGR

        if src_lnd_results is not None:
            
            src_lnds, src_poses, src_calib_lnds, src_idx_sets = src_lnd_results[0], src_lnd_results[1], src_lnd_results[2], src_lnd_results[3]
        
            if id_zs.shape[0] != src_lnds.shape[0]:
                # id_imgs = torch.cat([id_imgs[i][None,...] for i in src_idx_sets], 0)
                # id_zs = torch.cat([id_zs[i][None,...] for i in src_idx_sets], 0)
                # attr_imgs = torch.cat([attr_imgs[i][None,...] for i in src_idx_sets], 0)
                return None
            
            # Identity embedding
            id_embeds = self.id_encoder(id_imgs)

            if self.args.parameter_embedding:
                # Convert landmark range and eular angle range to (0, 1) correspondingly
                #norm_calib_lnds = src_calib_lnds.reshape(-1, 102) / (id_imgs.shape[-1] - 1) # (0, 255) -> (0, 1)
                #norm_poses = (src_poses / 1.5708 + 1) / 2 # (-0.5pi, 0.5pi) -> (0, 1)

                attr_embeds = self.attr_encoder(attr_imgs)

                pose_control_vectors = self.pose_parameter_decoder(attr_embeds)
                expression_control_vectors = self.expression_parameter_decoder(attr_embeds)
                control_vector = torch.cat([pose_control_vectors, expression_control_vectors, self.style_padding], -1)
                # ortho_regularizer = self.pose_basis[0] @ self.pose_basis[1] + self.pose_basis[1] @ self.pose_basis[2] \
                #                     + self.pose_basis[0] @ self.pose_basis[2]
            else:
                # Attribute embedding
                attr_embeds = self.attr_encoder(attr_imgs)
                control_vector = self.reference_network(attr_embeds)

                control_vector = torch.cat([self.style_padding1, control_vector, self.style_padding2], -1)

            gen_imgs = self.stylegan_generator(id_zs, control_vector)

            mod_styles = []
            for block in self.stylegan_generator.synthesis.children():
                for layer in block.children():
                    if not isinstance(layer, networks_stylegan2.ToRGBLayer):
                        mod_styles.append(layer.modified_styles.squeeze())

            mod_styles = torch.cat(mod_styles, -1)

            return gen_imgs, id_embeds, src_lnds, src_poses, src_calib_lnds, src_idx_sets, mod_styles
        
        else:
            return None

    def _train(self):
        self.attr_encoder._train()
        if self.args.parameter_embedding:
            # self.reference_pose_encoder._train()
            # self.reference_pose_decoder._train()
            # self.reference_expression_encoder._train()
            # self.reference_expression_decoder._train()
            self.pose_parameter_decoder._train()
            self.expression_parameter_decoder._train()
        else:
            self.reference_network._train()

    def _test(self):
        self.attr_encoder._test()
        if self.args.parameter_embedding:
            # self.reference_pose_encoder._test()
            # self.reference_pose_decoder._test()
            # self.reference_expression_encoder._test()
            # self.reference_expression_decoder._test()
            self.pose_parameter_decoder._test()
            self.expression_parameter_decoder._test()
        else:
            self.reference_network._test()

    def _save(self, reason):
        self.attr_encoder._save(reason)
        if self.args.parameter_embedding:
            # self.reference_pose_encoder._save("_pose"+reason)
            # self.reference_pose_decoder._save("_pose"+reason)
            # self.reference_expression_encoder._save("_expression"+reason)
            # self.reference_expression_decoder._save("_expression"+reason)
            self.pose_parameter_decoder._save("_pose"+reason)
            self.expression_parameter_decoder._save("_expression"+reason)
        else:
            self.reference_network._save(reason)

    def _load(self, reason):
        self.attr_encoder._load(reason)
        if self.args.parameter_embedding:
            # self.reference_pose_encoder._load("_pose"+reason)
            # self.reference_pose_decoder._load("_pose"+reason)
            # self.reference_expression_encoder._load("_expression"+reason)
            # self.reference_expression_decoder._load("_expression"+reason)
            self.pose_parameter_decoder._load("_pose"+reason)
            self.expression_parameter_decoder._load("_expression"+reason)
        else:
            self.reference_network._load(reason)
