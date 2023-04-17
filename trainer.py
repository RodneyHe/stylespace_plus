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

import PIL.Image as Image

class Trainer(object):
    def __init__(self, args, model, device):
        self.args = args
        self.device = device
        self.model = model
        
        # Dataset
        train_dataset = GeneratedDataset(args, "train")
        train_id_sampler, train_attr_sampler, self.train_dataset_length = self.get_id_attr_sampler(len(train_dataset), args.cross_frequency)
        self.train_id_loader = DataLoader(train_dataset, batch_size=args.batch_size, sampler=train_id_sampler)
        self.train_attr_loader = DataLoader(train_dataset, batch_size=args.batch_size, sampler=train_attr_sampler)
        
        validate_dataset = GeneratedDataset(args, "validate")
        validate_id_sampler, validate_attr_sampler, self.validate_dataset_length = self.get_id_attr_sampler(len(validate_dataset), args.cross_frequency)
        self.validate_id_loader = DataLoader(validate_dataset, batch_size=args.batch_size, sampler=validate_id_sampler)
        self.validate_attr_loader = DataLoader(validate_dataset, batch_size=args.batch_size, sampler=validate_attr_sampler)

        # lrs & optimizers
        self.embedding_optimizer = torch.optim.Adam([p for p in list(self.model.parameters()) if p.requires_grad], lr=5e-6)
        self.regulizing_optimizer = torch.optim.Adam([p for p in list(self.model.parameters()) if p.requires_grad], lr=5e-8)

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
        self.lambda_pixel = 0.02

        self.lambda_id = 100
        self.lambda_landmarks = 0.001

        # Test
        self.test_not_imporved = 0
        self.max_id_preserve = 0
        self.min_lnd_dist = np.inf

    def train(self):
        # if self.phase == "regulizing":
        #     self.model._load(self.args.name + "/weights")
        
        print(f"Epoch: {self.num_epoch}")

        while self.num_epoch < self.args.num_epochs:

            try:

                # Train epoch
                self.train_epoch(phase="embedding")
                
                # Test epoch
                self.test_epoch(phase="embedding")

            except Exception as e:
                raise

            # if self.test_not_imporved > self.args.not_improved_exit:
            #    self.logger.info(f'Test has not improved for {self.args.not_improved_exit} epochs. Exiting...')
            #    break

            # if self.num_epoch == self.args.num_epochs:
            #     if self.phase == "embedding":
            #         self.model.my_save()
            #         break

            #     elif self.phase == "regulizing":
            #         self.model.my_save()
            #         break
            
            if self.num_epoch == self.args.num_epochs:
                self.model._save()

            self.num_epoch += 1

    # Train function
    def train_epoch(self, phase):
        self.model._train()
        # if phase == "embedding":
        #     self.model._train()
        # elif phase == "regularizing":
        #     pass

        pbar = tqdm(range(self.train_dataset_length), ncols=60)
        for step in pbar:
            if self.args.cross_frequency and (step % self.args.cross_frequency == 0):
                self.is_cross_epoch = True
            else:
                self.is_cross_epoch = False
            
            id_images, id_zs = next(iter(self.train_id_loader))
            id_images = id_images.to(self.device)
            id_zs = id_zs.to(self.device)
            
            if self.is_cross_epoch:
                attr_images, _ = next(iter(self.train_attr_loader))
                attr_images = attr_images.to(self.device)
            else:
                attr_images = id_images

            gen_images, id_embedding, _, src_landmarks, src_idx_list = self.model.generator(id_images, id_zs, attr_images)
            
            if src_landmarks is None:
                continue

            if self.args.id_loss:
                gen_id_embedding = self.model.generator.id_encoder(gen_images)
                id_loss = self.lambda_id * self.id_loss_func(gen_id_embedding, id_embedding)
            
            if self.args.landmarks_loss:
                try:
                    pred_landmarks, pred_idx_list = self.model.generator.landmarks_detector(gen_images)
                except Exception as e:
                    pred_landmarks = None

                if len(src_idx_list) != len(pred_idx_list) or pred_landmarks is None:
                    landmarks_loss = 0
                else:
                    landmarks_loss = self.lambda_landmarks * torch.mean(self.landmarks_loss_fun(pred_landmarks, src_landmarks))

            if not self.is_cross_epoch and self.args.pixel_loss:
                l1_loss = torch.mean(self.pixel_loss_func(gen_images, id_images) * self.pixel_mask)

                if self.args.pixel_loss_type == 'mix':
                    mssim = torch.mean(1 - ms_ssim_metric(attr_images, gen_images, data_range=1.0))
                    pixel_loss = self.lambda_pixel * (0.84 * mssim + 0.16 * l1_loss)
                else:
                    pixel_loss = self.lambda_pixel * l1_loss
            else:
                pixel_loss = 0

            total_loss = id_loss + landmarks_loss + pixel_loss

            Writer.add_scalar("loss/id_loss", id_loss, step=(step+self.num_epoch*self.train_dataset_length))
            Writer.add_scalar("loss/landmarks_loss", landmarks_loss, step=(step+self.num_epoch*self.train_dataset_length))
            Writer.add_scalar("loss/total_loss", total_loss, step=(step+self.num_epoch*self.train_dataset_length))

            if not self.is_cross_epoch:
                Writer.add_scalar("loss/pixel_loss", pixel_loss, step=(step+self.num_epoch*self.train_dataset_length))

            if step % 3000 == 0 and self.is_cross_epoch:
                general_utils.save_images(id_images, attr_images, gen_images, src_landmarks, pred_landmarks, 256,
                                          self.args.images_results.joinpath(f"e{self.num_epoch}_s{step}_.png"), 3)

            if total_loss != 0:
                self.embedding_optimizer.zero_grad()
                total_loss.backward()
                self.embedding_optimizer.step()

    # Test function
    def test_epoch(self):
        self.model._test()

        similarities = {'id_to_pred': [], 'id_to_attr': [], 'attr_to_pred': []}

        fake_reconstruction = {'MSE': [], 'PSNR': [], 'ID': []}
        real_reconstruction = {'MSE': [], 'PSNR': [], 'ID': []}

        lnd_dist = []
        for i in range(self.args.test_size):
            id_images, id_zs = next(iter(self.validate_id_loader))
            id_images = id_images.to(self.device)
            id_zs = id_zs.to(self.device)
            attr_images, attr_zs = next(iter(self.validate_attr_loader))
            attr_images = attr_images.to(self.device)

            gen_images, id_embedding, attr_embedding, src_landmarks, _ = self.model.generator(id_images, id_zs, attr_images)
            
            if src_landmarks is None:
                continue
            
            gen_id_embedding = self.model.generator.id_encoder(gen_images)
            
            similarities['id_to_pred'].extend(nn.functional.cosine_similarity(id_embedding, gen_id_embedding).numpy())
            similarities['id_to_attr'].extend(nn.functional.cosine_similarity(id_embedding, attr_embedding).numpy())
            similarities['attr_to_pred'].extend(nn.functional.cosine_similarity(attr_embedding, gen_id_embedding).numpy())

            # Landmarks
            try:
                pred_landmarks, _ = self.model.generator.landmarks_detector(gen_images)
            except Exception as e:
                pred_landmarks = None

            if pred_landmarks is None:
                continue

            lnd_dist.extend(nn.functional.mse_loss(src_landmarks, pred_landmarks).numpy())

            # Fake reconstruction (using generated image as attribute image)
            self.test_reconstruction(id_images, id_zs, fake_reconstruction, display=(i==0), display_name='id_img')

            if self.args.test_real_attr:
                # Real Reconstruction (using real image as attribute image)
                self.test_reconstruction(attr_images, attr_zs, real_reconstruction, display=(i==0), display_name='attr_img')

            if i == 0:
                general_utils.save_images(id_images, attr_images, gen_images, src_landmarks, pred_landmarks, 256,
                                          self.args.images_results.joinpath(f"test_e{self.num_epoch}.png"), 3)

                Writer.add_image('test/prediction', [id_images[0], attr_images[0], gen_images[0]], step=self.num_epoch)

        # Similarity
        mean_lnd_dist = np.mean(lnd_dist)

        id_to_pred = np.mean(similarities['id_to_pred'])
        attr_to_pred = np.mean(similarities['attr_to_pred'])
        mean_disen = attr_to_pred - id_to_pred

        Writer.add_scalar('similarity/score', mean_disen, step=self.num_epoch)
        Writer.add_scalar('similarity/id_to_pred', id_to_pred, step=self.num_epoch)
        Writer.add_scalar('similarity/attr_to_pred', attr_to_pred, step=self.num_epoch)

        Writer.add_scalar('landmarks/L2', np.mean(lnd_dist), step=self.num_epoch)

        # Reconstruction
        # if self.args.test_real_attr:
        #     Writer.add_scalar('reconstruction/real_MSE', np.mean(real_reconstruction['MSE']), step=self.num_epoch)
        #     Writer.add_scalar('reconstruction/real_PSNR', np.mean(real_reconstruction['PSNR']), step=self.num_epoch)
        #     Writer.add_scalar('reconstruction/real_ID', np.mean(real_reconstruction['ID']), step=self.num_epoch)

        Writer.add_scalar('reconstruction/fake_MSE', np.mean(fake_reconstruction['MSE']), step=self.num_epoch)
        Writer.add_scalar('reconstruction/fake_PSNR', np.mean(fake_reconstruction['PSNR']), step=self.num_epoch)
        Writer.add_scalar('reconstruction/fake_ID', np.mean(fake_reconstruction['ID']), step=self.num_epoch)

        # if mean_lnd_dist < self.min_lnd_dist:
        #     self.logger.info('Minimum landmarks dist achieved. saving checkpoint')
        #     self.test_not_imporved = 0
        #     self.min_lnd_dist = mean_lnd_dist
        #     self.model.my_save(f'_best_landmarks_epoch_{self.num_epoch}')

        # if np.abs(id_to_pred) > self.max_id_preserve:
        #     self.logger.info(f'Max ID preservation achieved! saving checkpoint')
        #     self.test_not_imporved = 0
        #     self.max_id_preserve = np.abs(id_to_pred)
        #     self.model.my_save(f'_best_id_epoch_{self.num_epoch}')

        # else:
        #     self.test_not_imporved += 1

    # Helper function
    def test_reconstruction(self, images, zs_matching, errors_dict, display=False, display_name=None):

        gen_images, id_embedding, attr_embedding, src_landmarks, idx_list = self.model.generator(images, zs_matching, images)

        recon_image = ((gen_images.flip(-3) + 1) / 2).clamp(0, 1)
        recon_gen_id_embedding = self.model.generator.id_encoder(recon_image)

        mse = nn.functional.mse_loss(images, recon_image).numpy()
        psnr = psnr_metric(recon_image, recon_image, data_range=1.0).numpy()

        errors_dict['MSE'].extend(mse)
        errors_dict['PSNR'].extend(psnr)
        errors_dict['ID'].extend(nn.functional.cosine_similarity(id_embedding, recon_gen_id_embedding).numpy())

        if display:
            Writer.add_image(f'reconstruction/input_{display_name}_img', images, step=self.num_epoch)
            Writer.add_image(f'reconstruction/pred_{display_name}_img', gen_images, step=self.num_epoch)

    def get_id_attr_sampler(self, dataset_length, cross_frequency):
        split = dataset_length // (cross_frequency + 1)
        id_dataset_length = dataset_length - split
        indices = list(range(dataset_length))
        random.shuffle(indices)
        id_sampler =  SubsetRandomSampler(indices[split:])
        attr_sampler = SubsetRandomSampler(indices[:split])
        return id_sampler, attr_sampler, id_dataset_length
