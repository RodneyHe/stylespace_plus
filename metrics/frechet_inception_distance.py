import torch, torchvision, os, uuid, copy
import torch.nn as nn
from torch.utils.data.dataloader import DataLoader
import torchvision.transforms.functional as TF
import numpy as np
import scipy.linalg
from data_loader.ffhq_data import FFHQDataset
from data_loader.gen_data import GeneratedDataset
from . import metric_utils

from tqdm import tqdm

class FIDScore(object):
    def __init__(self, args, G, device) -> None:
        self.args = args
        self.device = device
        self.G = G

        # Feature detector
        self.inception_network = torchvision.models.inception_v3(weights=torchvision.models.Inception_V3_Weights.DEFAULT)
        self.inception_network.aux_logits = False
        self.inception_network.dropout = nn.Identity()
        self.inception_network.fc = nn.Identity()
        self.inception_network.eval().to(device)

        # Generated dataset
        self.gen_dataset = GeneratedDataset(args)
        self.gen_dataset_dir = os.path.join(args.dataset_path, "gen_dataset")

        # Real dataset
        self.ffhq_dataset = FFHQDataset(self.args)
        self.real_dataset_dir = os.path.join(args.dataset_path, "ffhq256_dataset")

    def calculate_feature_for_dataset(self, max_real=None):
        # Try to lookup from cache
        cache_file = None
        if self.args.cache:
            cache_file = os.path.join(self.real_dataset_dir, "ffhq_feature_cache.pkl")
            #cache_file = os.path.join(self.gen_dataset_dir, "gen_feature_cache.pkl")

            # Load
            if os.path.isfile(cache_file):
                return metric_utils.FeatureStats.load(cache_file)

        # Initialize
        stats = metric_utils.FeatureStats(capture_mean_cov=True, max_items=max_real)
        ffhq_dataloader = DataLoader(self.ffhq_dataset, batch_size=6, pin_memory=True)
        ffhq_iter = iter(ffhq_dataloader)
        # gen_dataloader = DataLoader(self.gen_dataset, batch_size=1, pin_memory=True)
        # gen_iter = iter(gen_dataloader)

        pbar = tqdm(range(len(ffhq_iter)), ncols=80)
        for _ in pbar:
            images = next(ffhq_iter)
            images = ((images + 1) / 2).clamp(0, 1).to(self.device)
            images = TF.resize(images, (299, 299), antialias=True)

            with torch.no_grad():
                features = self.inception_network(images)
            
            stats.append_torch(features)

            if stats.is_full():
                break
        
        # Save to cache
        if cache_file is not None:
            os.makedirs(os.path.dirname(cache_file), exist_ok=True)
            temp_file = cache_file + "." + uuid.uuid4().hex
            stats.save(temp_file)
            os.replace(temp_file, cache_file)
        return stats
    
    def calculate_feature_for_generator(self, batch_size=64, batch_gen=None, num_gen=None):

        # Setup generator
        G = copy.deepcopy(self.G).eval().requires_grad_(False).to(self.device)

        # Initialize
        ffhq_dataloader = DataLoader(self.ffhq_dataset, batch_size=batch_size, shuffle=True, drop_last=True, pin_memory=True)
        ffhq_iter = iter(ffhq_dataloader)

        if num_gen is None:
            num_gen = len(ffhq_iter) * batch_size
            
        stats = metric_utils.FeatureStats(capture_mean_cov=True, max_items=num_gen)

        style_padding = torch.zeros((batch_size, 2368)).to(self.device)
        self.style_padding1 = torch.zeros((batch_size, 1536), requires_grad=False).to(self.device)
        self.style_padding2 = torch.zeros((batch_size, 1344), requires_grad=False).to(self.device)

        # Generate samples
        while not stats.is_full():
            z = torch.randn((batch_size, 512)).to(self.device)
            ws = G.stylegan_generator.mapping(z, 0)
            #ctrl_vecs = torch.zeros((batch_size, 4928)).to(self.device)
            
            #id_images = G.stylegan_generator.synthesis(ws, ctrl_vecs)

            attr_images_from_real = next(ffhq_iter)
            with torch.no_grad():
                attr_embeds = G.attr_encoder(attr_images_from_real.to(self.device))
                #id_embeds = G.id_encoder(id_images)
                ctrl_vecs = G.reference_network(attr_embeds)
                #ctrl_vecs = torch.cat([ctrl_vecs, style_padding], -1)
                ctrl_vecs = torch.cat([self.style_padding1, ctrl_vecs, self.style_padding2], -1)

                gen_images = G.stylegan_generator.synthesis(ws, ctrl_vecs)

                gen_images = ((gen_images + 1) / 2).clamp(0, 1)
                gen_images = TF.resize(gen_images, (299, 299), antialias=True)
                features = self.inception_network(gen_images)
        
            stats.append_torch(features)
        
        return stats
        
    def calculate_fid(self, max_real=None, num_gen=None):
        mu_real, sigma_real = self.calculate_feature_for_dataset(max_real=max_real).get_mean_cov()
        mu_gen, sigma_gen = self.calculate_feature_for_generator(num_gen=num_gen).get_mean_cov()

        m = np.square(mu_gen - mu_real).sum()
        s, _ = scipy.linalg.sqrtm(np.dot(sigma_gen, sigma_real), disp=False) # pylint: disable=no-member
        fid = np.real(m + np.trace(sigma_gen + sigma_real - s * 2))
        return float(fid)