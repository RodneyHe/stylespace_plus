from tqdm import tqdm
import torch, random
import torch.nn as nn
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from torchmetrics.functional import multiscale_structural_similarity_index_measure as ms_ssim_metric, peak_signal_noise_ratio as psnr_metric

import numpy as np
from writer import Writer
from general_utils import general_utils
from data_loader.gen_data import GeneratedDataset

class Trainer(object):
    def __init__(self, args, model, device):
        self.args = args
        self.device = device
        self.model = model
        
        # Dataset & dataloader
        self.train_dataset = GeneratedDataset(args, "train")
        self.validate_dataset = GeneratedDataset(self.args, "validate")
        
        train_id_sampler, train_attr_sampler = self.get_id_attr_sampler(len(self.train_dataset), self.args.cross_frequency)
        self.train_id_loader = DataLoader(self.train_dataset, batch_size=self.args.batch_size, sampler=train_id_sampler, pin_memory=True)
        self.train_attr_loader = DataLoader(self.train_dataset, batch_size=self.args.batch_size, sampler=train_attr_sampler, pin_memory=True)
        
        validate_id_sampler, validate_attr_sampler = self.get_id_attr_sampler(len(self.validate_dataset), self.args.cross_frequency)
        self.validate_id_loader = DataLoader(self.validate_dataset, batch_size=self.args.batch_size, sampler=validate_id_sampler, pin_memory=True)
        self.validate_attr_loader = DataLoader(self.validate_dataset, batch_size=self.args.batch_size, sampler=validate_attr_sampler, pin_memory=True)
        
        
        


        

        # Dataset statistics
        self.styles_mean = torch.from_numpy(np.load(args.dataset_path.joinpath("gen_dataset/styles_mean.npy"))).to(device)
        self.styles_std = torch.from_numpy(np.load(args.dataset_path.joinpath("gen_dataset/styles_std.npy"))).to(device)

        # lrs & optimizers
        self.embedding_optimizer = torch.optim.Adam(self.model.parameters(), lr=5e-7, weight_decay=5e-4)
        self.regulizing_optimizer = torch.optim.Adam(self.model.parameters(), lr=5e-8, weight_decay=5e-4)

        # Losses
        self.id_loss_func = nn.L1Loss()
        self.landmarks_loss_fun = nn.MSELoss()
        self.pixel_loss_func = nn.L1Loss(reduce=None)

        if args.pixel_mask_type == 'gaussian':
            sigma = int(80 * (self.args.resolution / 256))
            self.pixel_mask = general_utils.inverse_gaussian_image(self.args.resolution, sigma).to(self.device)
        else:
            self.pixel_mask = torch.ones([self.args.resolution, self.args.resolution]).to(self.device)
            self.pixel_mask = self.pixel_mask / torch.sum(self.pixel_mask)

        self.pixel_mask = torch.broadcast_to(self.pixel_mask, [self.args.batch_size, *self.pixel_mask.shape])

        self.num_epoch = 0
        self.is_cross_epoch = False
        #self.phase = self.model.phase

        # Lambdas
        self.lambda_id = 1
        self.lambda_landmarks = 0.01
        self.lambda_pixel = 0.02
        self.lambda_styles_regularizer = 5e-8

        # Test
        self.test_not_imporved = 0
        self.max_id_preserve = 0
        self.min_lnd_dist = np.inf

    def train(self):
        # if self.phase == "regulizing":
        #     self.model._load(self.args.name + "/weights")

        while True:
            
            print(f"Epoch: {self.num_epoch}")

            try:
                # Train epoch
                self.train_epoch(phase="embedding")
                
                # Test epoch
                with torch.no_grad():
                    self.test_epoch(phase="embedding")

            except Exception as e:
                raise

            # if self.test_not_imporved > self.args.not_improved_exit:
            #    print(f"Test has not improved for {self.args.not_improved_exit} epochs. Exiting...")
            #    break

            # if self.num_epoch == self.args.num_epochs:
            #     if self.phase == "embedding":
            #         self.model.my_save()
            #         break

            #     elif self.phase == "regulizing":
            #         self.model.my_save()
            #         break
            
            if self.num_epoch == self.args.num_epochs - 1:
                self.model._save()
                break

            self.num_epoch += 1

    # Train function
    def train_epoch(self, phase):
        # if phase == "embedding":
        #     self.model._train()
        # elif phase == "regularizing":
        #     pass
        
        self.train_id_iter = iter(self.train_id_loader)
        self.train_attr_iter = iter(self.train_attr_loader)

        self.validate_id_iter = iter(self.validate_id_loader)
        self.validate_attr_iter = iter(self.validate_attr_loader)

        pbar = tqdm(range(len(self.train_id_iter)), ncols=80, postfix="train_epoch")
        for step in pbar:
            if self.args.cross_frequency and ((step + 1) % self.args.cross_frequency == 0):
                self.is_cross_epoch = True
            else:
                self.is_cross_epoch = False
            
            id_images, id_zs = next(self.train_id_iter)
            id_images = id_images.to(self.device)
            id_zs = id_zs.to(self.device)
            
            if self.is_cross_epoch:
                attr_images, _ = next(self.train_attr_iter)
                attr_images = attr_images.to(self.device)
            else:
                attr_images = id_images

            gen_images, id_embedding, _, src_landmarks, src_pose, src_idx_list, modified_styles = self.model.generator(id_images, id_zs, attr_images)
            
            # Compute the negative log likelihood of style space
            #styles_regularizer = self.lambda_styles_regularizer * torch.pow((modified_styles-torch.broadcast_to(self.styles_mean, (modified_styles.shape[0], modified_styles.shape[1]))), 2).sum()
            
            if src_landmarks is None:
                continue

            if self.args.id_loss:
                gen_id_embedding = self.model.generator.id_encoder(gen_images)
                id_loss = self.lambda_id * self.id_loss_func(gen_id_embedding, id_embedding)
            
            if self.args.landmarks_loss:
                
                pred_landmarks, pred_pose, pred_idx_list = self.model.generator.landmarks_detector(gen_images.flip(-3)) # the input is converted to BGR

                if pred_landmarks is None or len(src_idx_list) != len(pred_idx_list):
                    continue
                else:
                    landmarks_loss = self.lambda_landmarks * (torch.mean(self.landmarks_loss_fun(pred_landmarks, src_landmarks)) \
                                      + torch.mean(self.landmarks_loss_fun(pred_pose, src_pose)))
                if landmarks_loss > 1:
                    continue

            if not self.is_cross_epoch and self.args.pixel_loss:
                l1_loss = torch.mean(self.pixel_loss_func(attr_images, gen_images) * self.pixel_mask)

                if self.args.pixel_loss_type == 'mix':
                    mssim = torch.mean(1 - ms_ssim_metric(attr_images, gen_images, data_range=1.0))
                    pixel_loss = self.lambda_pixel * (0.84 * mssim + 0.16 * l1_loss)
                else:
                    pixel_loss = self.lambda_pixel * l1_loss
            else:
                pixel_loss = 0

            total_loss = id_loss + landmarks_loss + pixel_loss

            Writer.add_scalar("loss/id_loss", id_loss, step=(step+self.num_epoch*len(self.train_id_iter)))
            Writer.add_scalar("loss/landmarks_loss", landmarks_loss, step=(step+self.num_epoch*len(self.train_id_iter)))
            #Writer.add_scalar("loss/styles_regularizer", styles_regularizer, step=(step+self.num_epoch*len(self.train_id_iter)))
            Writer.add_scalar("loss/total_loss", total_loss, step=(step+self.num_epoch*len(self.train_id_iter)))

            if not self.is_cross_epoch:
                Writer.add_scalar("loss/pixel_loss", pixel_loss, step=(step+self.num_epoch*len(self.train_id_iter)))

            if (step + 1) % 3000 == 0 and self.is_cross_epoch:
                general_utils.save_images(id_images, 
                                          attr_images, 
                                          gen_images, 
                                          self.args.images_results.joinpath(f"e{self.num_epoch}_s{step}_.png"), landmarks=False)

            if total_loss != 0:
                self.embedding_optimizer.zero_grad()
                total_loss.backward()
                self.embedding_optimizer.step()
        
            # Test process
            similarities = {"id_to_pred": [], "id_to_attr": [], "attr_to_pred": []}

            id_images, id_zs = next(iter(self.validate_id_loader))
            id_images = id_images.to(self.device)
            id_zs = id_zs.to(self.device)
            attr_images, attr_zs = next(self.validate_attr_iter)
            attr_images = attr_images.to(self.device)

            gen_images, id_embedding, _, src_landmarks, src_pose, src_idx_list, _ = self.model.generator(id_images, id_zs, attr_images)

            if src_landmarks is None:
                continue

            gen_id_embedding = self.model.generator.id_encoder(gen_images)
            attr_id_embedding = self.model.generator.id_encoder(attr_images)

    # Test function
    def test_batch(self, phase, save_image=True):

        similarities = {"id_to_pred": [], "id_to_attr": [], "attr_to_pred": []}

        fake_reconstruction = {"MSE": [], "PSNR": [], "ID": []}
        real_reconstruction = {"MSE": [], "PSNR": [], "ID": []}
        
        self.validate_id_iter = iter(self.validate_id_loader)
        self.validate_attr_iter = iter(self.validate_attr_loader)

        lnd_dist = []
        pose_dist = []
        save_image = True
        pbar = tqdm(range(self.args.test_size), ncols=80, postfix="test_epoch")
        for step in pbar:
            id_images, id_zs = next(iter(self.validate_id_loader))
            id_images = id_images.to(self.device)
            id_zs = id_zs.to(self.device)
            attr_images, attr_zs = next(self.validate_attr_iter)
            attr_images = attr_images.to(self.device)

            gen_images, id_embedding, _, src_landmarks, src_pose, src_idx_list, _ = self.model.generator(id_images, id_zs, attr_images)
            
            if src_landmarks is None:
                continue

            gen_id_embedding = self.model.generator.id_encoder(gen_images)
            attr_id_embedding = self.model.generator.id_encoder(attr_images)

            similarities['id_to_pred'].append(nn.functional.cosine_similarity(id_embedding, gen_id_embedding).mean().item())
            similarities['id_to_attr'].append(nn.functional.cosine_similarity(id_embedding, attr_id_embedding).mean().item())
            similarities['attr_to_pred'].append(nn.functional.cosine_similarity(attr_id_embedding, gen_id_embedding).mean().item())

            # Landmarks
            pred_landmarks, pred_pose, pred_idx_list = self.model.generator.landmarks_detector(gen_images.flip(-3)) # the input is converted to BGR

            if pred_landmarks is None or len(src_idx_list) != len(pred_idx_list):
                continue
            else:
                lnd_dist.append(nn.functional.mse_loss(src_landmarks, pred_landmarks).item())
                pose_dist.append(nn.functional.mse_loss(src_pose, pred_pose).item())

            # Fake reconstruction (using generated image as attribute image)
            self.test_reconstruction(id_images, id_zs, fake_reconstruction, display=(step==0), display_name="id_img")

            if self.args.test_real_attr:
                # Real Reconstruction (using real image as attribute image)
                self.test_reconstruction(attr_images, attr_zs, real_reconstruction, display=(step==0), display_name="attr_img")

            if save_image:
                general_utils.save_images(id_images, 
                                          attr_images, 
                                          gen_images, 
                                          self.args.images_results.joinpath(f"test_e{self.num_epoch}.png"),
                                          sample_number=3,
                                          landmarks=True,
                                          attr_landmarks=src_landmarks[:,:2,:],
                                          gen_landmarks=pred_landmarks[:,:2,:])

                Writer.add_image("test/prediction", [attr_images[0], gen_images[0], id_images[0]], step=self.num_epoch)
                save_image = False

        # Similarity
        mean_lnd_dist = np.mean(lnd_dist)

        id_to_pred = np.mean(similarities["id_to_pred"])
        attr_to_pred = np.mean(similarities["attr_to_pred"])
        mean_disen = attr_to_pred - id_to_pred

        Writer.add_scalar("similarity/score", mean_disen, step=self.num_epoch)
        Writer.add_scalar("similarity/id_to_pred", id_to_pred, step=self.num_epoch)
        Writer.add_scalar("similarity/attr_to_pred", attr_to_pred, step=self.num_epoch)

        Writer.add_scalar("landmarks/L2", np.mean(lnd_dist), step=self.num_epoch)
        Writer.add_scalar("poses/L2", np.mean(pose_dist), step=self.num_epoch)

        # Reconstruction
        # if self.args.test_real_attr:
        #     Writer.add_scalar('reconstruction/real_MSE', np.mean(real_reconstruction['MSE']), step=self.num_epoch)
        #     Writer.add_scalar('reconstruction/real_PSNR', np.mean(real_reconstruction['PSNR']), step=self.num_epoch)
        #     Writer.add_scalar('reconstruction/real_ID', np.mean(real_reconstruction['ID']), step=self.num_epoch)

        Writer.add_scalar("reconstruction/fake_MSE", np.mean(fake_reconstruction['MSE']), step=self.num_epoch)
        Writer.add_scalar("reconstruction/fake_PSNR", np.mean(fake_reconstruction['PSNR']), step=self.num_epoch)
        Writer.add_scalar("reconstruction/fake_ID", np.mean(fake_reconstruction['ID']), step=self.num_epoch)

        if mean_lnd_dist < self.min_lnd_dist:
            print("Minimum landmarks dist achieved. Saving checkpoint")
            self.test_not_imporved = 0
            self.min_lnd_dist = mean_lnd_dist
            self.model._save(f"_best_landmarks")

        if np.abs(id_to_pred) > self.max_id_preserve:
            print("Max ID preservation achieved! saving checkpoint")
            self.test_not_imporved = 0
            self.max_id_preserve = np.abs(id_to_pred)
            self.model._save(f"_best_id")
        else:
            self.test_not_imporved += 1

    # Helper function
    def test_reconstruction(self, images, zs_matching, errors_dict, display=False, display_name=None):

        recon_images, id_embedding, _, _, _, _, _ = self.model.generator(images, zs_matching, images)
        recon_gen_id_embedding = self.model.generator.id_encoder(recon_images)

        mse = nn.functional.mse_loss(images, recon_images).item()
        psnr = psnr_metric(recon_images, images, data_range=1.0).item()

        errors_dict["MSE"].append(mse)
        errors_dict["PSNR"].append(psnr)
        errors_dict["ID"].append(nn.functional.cosine_similarity(id_embedding, recon_gen_id_embedding).mean().item())

        if display:
            Writer.add_image(f"reconstruction/input_{display_name}_img", [images[0], recon_images[0]], step=self.num_epoch)

    def get_id_attr_sampler(self, dataset_length, cross_frequency):
        split = dataset_length // (cross_frequency + 1)
        indices = list(range(dataset_length))
        random.shuffle(indices)
        id_sampler =  SubsetRandomSampler(indices[split:])
        attr_sampler = SubsetRandomSampler(indices[:split])
        return id_sampler, attr_sampler
