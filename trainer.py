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
from data_loader.ffhq_data import FFHQDataset
from metrics.frechet_inception_distance import FIDScore

import PIL.Image as Image

class Trainer(object):
    def __init__(self, args, model, device):
        self.args = args
        self.device = device
        self.model = model
        
        # Dataset & dataloader
        self.train_dataset = GeneratedDataset(args, "train")
        self.validate_dataset = GeneratedDataset(args, "validate")
        
        train_id_sampler, train_attr_sampler = self.get_id_attr_sampler(len(self.train_dataset), self.args.cross_frequency)
        self.train_id_loader = DataLoader(self.train_dataset, batch_size=self.args.batch_size, sampler=train_id_sampler, pin_memory=True)
        self.train_attr_loader = DataLoader(self.train_dataset, batch_size=self.args.batch_size, sampler=train_attr_sampler, pin_memory=True)
        
        validate_id_sampler, validate_attr_sampler = self.get_id_attr_sampler(len(self.validate_dataset), self.args.cross_frequency)
        self.validate_id_loader = DataLoader(self.validate_dataset, batch_size=self.args.batch_size, sampler=validate_id_sampler, pin_memory=True)
        self.validate_attr_loader = DataLoader(self.validate_dataset, batch_size=self.args.batch_size, sampler=validate_attr_sampler, pin_memory=True)

        if args.test_real_attr:
            self.real_attr_dataset = FFHQDataset(args)
            self.real_attr_dataloader = DataLoader(self.real_attr_dataset, batch_size=6, pin_memory=True)

        # Dataset statistics
        self.styles_mean = torch.from_numpy(np.load(args.dataset_path.joinpath("gen_dataset/styles_mean.npy"))).to(device)
        self.styles_std = torch.from_numpy(np.load(args.dataset_path.joinpath("gen_dataset/styles_std.npy"))).to(device)

        # lrs & optimizers
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=5e-6, weight_decay=5e-4)

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

        # Lambdas
        self.lambda_id = 1
        self.lambda_landmarks = 0.01
        self.lambda_pixel = 0.02
        self.lambda_styles_regularizer = 5e-7

        # Test
        self.test_not_imporved = 0
        self.max_id_preserve = 0
        self.min_lnd_dist = np.inf

    def train(self):
        
        # Main loop
        while True:
            print(f"Epoch: {self.num_epoch}")

            try:
                # Dynanmically change the learning rate
                if self.num_epoch >= 5:
                    self.adjust_learning_rate(self.optimizer, 1e-6)
                elif self.num_epoch >= 10:
                    self.adjust_learning_rate(self.optimizer, 5e-7)
                elif self.num_epoch >= 20:
                    self.adjust_learning_rate(self.optimizer, 1e-7)
                elif self.num_epoch >= 30:
                    self.adjust_learning_rate(self.optimizer, 5e-8)

                # Train epoch
                self.train_epoch()

                # Test epoch
                with torch.no_grad():
                    self.test_epoch()

            except Exception as e:
                raise

            # if self.test_not_imporved > self.args.not_improved_exit:
            #    print(f"Test has not improved for {self.args.not_improved_exit} epochs. Exiting...")
            #    break

            if self.num_epoch == self.args.num_epochs - 1:
                print(f"Training finished on {self.num_epoch} epoch. Exiting...")
                self.model._save()
                break

            self.num_epoch += 1

    # Train function
    def train_epoch(self):
        
        self.train_id_iter = iter(self.train_id_loader)
        self.train_attr_iter = iter(self.train_attr_loader)

        pbar = tqdm(range(len(self.train_id_iter)), ncols=80, postfix="train_epoch")
        for step in pbar:
            
            if self.args.cross_frequency and ((step + 1) % self.args.cross_frequency == 0):
                self.is_cross_epoch = True
            else:
                self.is_cross_epoch = False
            
            id_imgs, id_zs = next(self.train_id_iter)
            id_imgs = id_imgs.to(self.device)
            id_zs = id_zs.to(self.device)
            
            #print(self.args.cross_frequency and ((step + 1) % self.args.cross_frequency == 0), self.is_cross_epoch)
            
            if self.is_cross_epoch:
                attr_imgs, _ = next(self.train_attr_iter)
                attr_imgs = attr_imgs.to(self.device)
            else:
                attr_imgs = id_imgs
            
            src_results = self.model.generator(id_imgs, id_zs, attr_imgs)
            
            if src_results is None:
                continue

            gen_imgs, id_embeds, src_lnds, src_poses, src_calib_lnds, src_idx_sets, mod_styles = src_results[0], src_results[1], \
                src_results[2], src_results[3], src_results[4],  src_results[5], src_results[6]

            if self.args.id_loss:
                gen_id_embeds = self.model.generator.id_encoder(gen_imgs)
                id_loss = self.lambda_id * self.id_loss_func(gen_id_embeds, id_embeds)
            
            if self.args.landmarks_loss:
                pred_lnd_results = self.model.generator.landmarks_detector(gen_imgs.flip(-3)) # the input is converted to BGR
                
                if pred_lnd_results is None:
                    #print("pred_lnd_results is None")
                    continue
                
                pred_lnds, pred_poses, pred_calib_lnds, pred_idx_sets = pred_lnd_results[0], pred_lnd_results[1], pred_lnd_results[2], pred_lnd_results[3]
                
                # general_utils.save_images(id_images=id_imgs, 
                #                           attr_images=attr_imgs, 
                #                           gen_images=gen_imgs,
                #                           output_path=self.args.images_results.joinpath(f"e{self.num_epoch}_s{step}_test.png"), 
                #                           sample_number=3,
                #                           landmarks=True,
                #                           attr_landmarks=src_calib_lnds,
                #                           gen_landmarks=pred_calib_lnds)
                
                if len(src_idx_sets) != len(pred_idx_sets):
                    # intersets = src_idx_sets & pred_idx_sets
                    # src_lnds = torch.cat([src_lnds[i][None,...] for i in intersets])
                    # src_poses = torch.cat([src_poses[i][None,...] for i in intersets])
                    # pred_lnds = torch.cat([pred_lnds[i][None,...] for i in intersets])
                    # pred_poses = torch.cat([pred_poses[i][None,...] for i in intersets])
                    #print("src_idx_lists len != pred_idx_lists")
                    continue
                else:
                    landmarks_loss = self.lambda_landmarks * torch.mean(self.landmarks_loss_fun(pred_lnds, src_lnds))
                    #calib_landmarks_loss = self.lambda_landmarks * 0.1 * torch.mean(self.landmarks_loss_fun(pred_calib_lnds, src_calib_lnds))
                    pose_loss = 10 * torch.mean(self.landmarks_loss_fun(pred_poses, src_poses))
                    attr_loss = pose_loss + landmarks_loss
                if attr_loss > 1:
                    #print("landmarks_loss > 1")
                    continue
            
            if not self.is_cross_epoch and self.args.pixel_loss:
                l1_loss = torch.mean(self.pixel_loss_func(attr_imgs, gen_imgs) * self.pixel_mask)

                if self.args.pixel_loss_type == 'mix':
                    mssim = torch.mean(1 - ms_ssim_metric(attr_imgs, gen_imgs, data_range=1.0))
                    pixel_loss = self.lambda_pixel * (0.84 * mssim + 0.16 * l1_loss)
                else:
                    pixel_loss = self.lambda_pixel * l1_loss
            else:
                pixel_loss = 0

            # Compute the negative log likelihood of style space
            styles_regularizer = self.lambda_styles_regularizer * torch.pow((mod_styles-torch.broadcast_to(self.styles_mean, (mod_styles.shape[0], mod_styles.shape[1]))), 2).sum()

            total_loss = id_loss + attr_loss + pixel_loss + styles_regularizer

            Writer.add_scalar("loss/id_loss", id_loss, step=(step+self.num_epoch*len(self.train_id_iter)))
            Writer.add_scalar("loss/attr_loss", attr_loss, step=(step+self.num_epoch*len(self.train_id_iter)))
            Writer.add_scalar("loss/landmark_loss", landmarks_loss, step=(step+self.num_epoch*len(self.train_id_iter)))
            #Writer.add_scalar("loss/calib_landmark_loss", calib_landmarks_loss, step=(step+self.num_epoch*len(self.train_id_iter)))
            Writer.add_scalar("loss/pose_loss", pose_loss, step=(step+self.num_epoch*len(self.train_id_iter)))
            Writer.add_scalar("loss/total_loss", total_loss, step=(step+self.num_epoch*len(self.train_id_iter)))

            if not self.is_cross_epoch:
                Writer.add_scalar("loss/pixel_loss", pixel_loss, step=(step+self.num_epoch*len(self.train_id_iter)))

            #print(self.args.cross_frequency and ((step + 1) % self.args.cross_frequency == 0), self.is_cross_epoch)
            
            if (step + 1) % 3000 == 0 and self.is_cross_epoch:
                general_utils.save_images(id_images=id_imgs, 
                                          attr_images=attr_imgs, 
                                          gen_images=gen_imgs, 
                                          output_path=self.args.images_results.joinpath(f"e{self.num_epoch}_s{step}_.png"), landmarks=False)

            if total_loss != 0:
                self.optimizer.zero_grad()
                total_loss.backward()
                self.optimizer.step()

    # Test function
    def test_epoch(self, save_image=True):

        similarities = {"id_to_pred": [], "id_to_attr": [], "attr_to_pred": []}
        fake_reconstruction = {"MSE": [], "PSNR": [], "ID": []}
        
        self.validate_id_iter = iter(self.validate_id_loader)
        self.validate_attr_iter = iter(self.validate_attr_loader)
        if self.args.test_real_attr:
            self.real_attr_iter = iter(self.real_attr_dataloader)

        lnd_dist = []
        pose_dist = []
        save_image = True
        pbar = tqdm(range(self.args.test_size), ncols=80, postfix="test_epoch")
        for step in pbar:
            id_imgs, id_zs = next(self.validate_id_iter)
            id_imgs = id_imgs.to(self.device)
            id_zs = id_zs.to(self.device)

            if self.args.test_real_attr:
                attr_imgs = next(self.real_attr_iter)
                attr_imgs = attr_imgs.to(self.device)
            else:
                attr_imgs, _ = next(self.validate_attr_iter)
                attr_imgs = attr_imgs.to(self.device)

            gen_results = self.model.generator(id_imgs, id_zs, attr_imgs)
            
            if gen_results is None:
                continue

            gen_imgs, id_embeds, src_lnds, src_poses, src_calib_lnds, src_idx_lists = gen_results[0], gen_results[1], gen_results[2], gen_results[3], gen_results[4], gen_results[5]

            gen_id_embeds = self.model.generator.id_encoder(gen_imgs)
            attr_id_embeds = self.model.generator.id_encoder(attr_imgs)

            similarities['id_to_pred'].append(nn.functional.cosine_similarity(id_embeds, gen_id_embeds).mean().item())
            similarities['id_to_attr'].append(nn.functional.cosine_similarity(id_embeds, attr_id_embeds).mean().item())
            similarities['attr_to_pred'].append(nn.functional.cosine_similarity(attr_id_embeds, gen_id_embeds).mean().item())

            # Landmarks
            pred_lnd_results = self.model.generator.landmarks_detector(gen_imgs.flip(-3)) # the input is converted to BGR
            
            if pred_lnd_results is None:
                continue
            
            pred_lnds, pred_poses, pred_calib_lnds, pred_idx_lists = pred_lnd_results[0], pred_lnd_results[1], pred_lnd_results[2], pred_lnd_results[3]

            if len(src_idx_lists) != len(pred_idx_lists):
                continue
            else:
                pose_dist.append(nn.functional.mse_loss(src_poses, pred_poses).item())
                lnd_dist.append(nn.functional.mse_loss(pred_lnds, src_lnds).item())

                # Fake reconstruction (using generated image as attribute image)
                self.test_reconstruction(id_imgs, id_zs, fake_reconstruction, display=(step==0), display_name="id_img")

                if save_image:
                    general_utils.save_images(id_images=id_imgs, 
                                              attr_images=attr_imgs, 
                                              gen_images=gen_imgs, 
                                              output_path=self.args.images_results.joinpath(f"test_e{self.num_epoch}.png"),
                                              sample_number=3,
                                              landmarks=True,
                                              attr_landmarks=src_lnds,
                                              gen_landmarks=pred_lnds)

                    Writer.add_image("test/prediction", [attr_imgs[0], gen_imgs[0], id_imgs[0]], step=self.num_epoch)
                    save_image = False

        # Similarity
        mean_lnd_dist = np.mean(lnd_dist)

        id_to_pred = np.mean(similarities["id_to_pred"])
        attr_to_pred = np.mean(similarities["attr_to_pred"])
        mean_disen = attr_to_pred - id_to_pred

        Writer.add_scalar("similarity/score", mean_disen, step=self.num_epoch)
        Writer.add_scalar("similarity/id_to_pred", id_to_pred, step=self.num_epoch)
        Writer.add_scalar("similarity/attr_to_pred", attr_to_pred, step=self.num_epoch)

        if self.args.test_real_attr:
            Writer.add_scalar("similarity/attr_to_pred", attr_to_pred, step=self.num_epoch)

        Writer.add_scalar("test/landmarks_L2", np.mean(lnd_dist), step=self.num_epoch)
        Writer.add_scalar("test/poses_L2", np.mean(pose_dist), step=self.num_epoch)

        # Writer.add_scalar("reconstruction/fake_MSE", np.mean(fake_reconstruction['MSE']), step=self.num_epoch)
        # Writer.add_scalar("reconstruction/fake_PSNR", np.mean(fake_reconstruction['PSNR']), step=self.num_epoch)
        # Writer.add_scalar("reconstruction/fake_ID", np.mean(fake_reconstruction['ID']), step=self.num_epoch)

        # FID
        # if not self.args.parameter_embedding:
        #     fid_score = FIDScore(self.args, self.model.generator, self.device)
        #     fid = fid_score.calculate_fid(max_real=None, num_gen=3000)
        #     Writer.add_scalar("test/FID", fid, step=self.num_epoch)

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

        recon_results = self.model.generator(images, zs_matching, images)
        
        if recon_results is None:
            return
        
        recon_imgs, id_embedding = recon_results[0], recon_results[1]

        recon_gen_id_embedding = self.model.generator.id_encoder(recon_imgs)

        mse = nn.functional.mse_loss(images, recon_imgs).item()
        psnr = psnr_metric(recon_imgs, images, data_range=1.0).item()

        errors_dict["MSE"].append(mse)
        errors_dict["PSNR"].append(psnr)
        errors_dict["ID"].append(nn.functional.cosine_similarity(id_embedding, recon_gen_id_embedding).mean().item())

        if display:
            Writer.add_image(f"reconstruction/input_{display_name}_img", [images[0], recon_imgs[0]], step=self.num_epoch)

    def get_id_attr_sampler(self, dataset_length, cross_frequency):
        split = dataset_length // (cross_frequency + 1)
        indices = list(range(dataset_length))
        random.shuffle(indices)
        id_sampler =  SubsetRandomSampler(indices[split:])
        attr_sampler = SubsetRandomSampler(indices[:split])
        return id_sampler, attr_sampler
    
    def adjust_learning_rate(self, optimizer, lr):
        """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
        #lr = base_lr * (0.1 ** (epoch // 10))
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr