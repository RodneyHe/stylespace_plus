from tqdm import tqdm
import torch, random
import torch.nn as nn
import torchvision.transforms.functional as F
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from pytorch_msssim import ms_ssim

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

        # Load the reference image to establish the coordinates
        # Using the image 00000.png No.31 landmark point as the reference point
        # _ref_img = Image.open(args.dataset_path.joinpath("gen_dataset/image/1000/3.png"))
        # _ref_img = F.to_tensor(_ref_img)[None, ...]
        # _ref_landmarks = self.model.generator.landmarks_detector(_ref_img)

        # self.ref_origin = _ref_landmarks[:, 30]

    def train(self):
        # if self.phase == "regulizing":
        #     self.model._load(self.args.name + "/weights")
        
        print(f"Epoch: {self.num_epoch}")

        while self.num_epoch < self.args.num_epochs:

            try:
                # if self.num_epoch % self.args.test_frequency == 0:
                #     self.test_epoch()

                self.train_epoch()

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

    def train_epoch(self):
        self.model._train()

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

            # if self.phase == "embedding":
            #     gen_img, id_embedding, _, _ = self.model.G(id_img, id_z_matching, 
            #                                                attr_img, self.phase, self.ref_origin)
            # elif self.phase == "regulizing":
            #     gen_img, id_embedding, _, _ = self.model.G(id_img, id_z_matching, 
            #                                                attr_img, self.phase, self.ref_origin, self.orthogonal_basis)

            if self.args.id_loss:
                gen_images_embedding = self.model.generator.id_encoder(gen_images)
                id_loss = self.lambda_id * self.id_loss_func(gen_images_embedding, id_embedding)
            
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
                    mssim = torch.mean(1 - ms_ssim(attr_images, gen_images, 1.0))
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

            if step % 300 == 0 and self.is_cross_epoch:
                general_utils.save_images(id_images, attr_images, gen_images, src_landmarks, pred_landmarks, 256,
                                          self.args.images_results.joinpath(f"e{self.num_epoch}_s{step}_.png"), 3)

            if total_loss != 0:
                self.embedding_optimizer.zero_grad()
                total_loss.backward()
                self.embedding_optimizer.step()

    # Common
    # Test
    # def test_epoch(self):
    #     self.model._test()

    #     similarities = {'id_to_pred': [], 'id_to_attr': [], 'attr_to_pred': []}

    #     fake_reconstruction = {'MSE': [], 'PSNR': [], 'ID': []}
    #     #real_reconstruction = {'MSE': [], 'PSNR': [], 'ID': []}

    #     if self.args.test_with_arcface:
    #         test_similarities = {'id_to_pred': [], 'id_to_attr': [], 'attr_to_pred': []}

    #     lnd_dist = []
    #     for i in range(self.args.test_size):
    #         id_img, id_z_matching, attr_img, _ = self.data_loader.get_batch(is_train=False, is_cross=True)

    #         gen_img, id_embedding, _, attr_lnds = self.model.G(id_img, id_z_matching, attr_img, self.ref_origin)

    #         gen_img_embedding = self.model.G.id_encoder(gen_img)
    #         attr_img_id_embedding = self.model.G.id_encoder(attr_img)

    #         similarities['id_to_pred'].extend(tf.keras.losses.cosine_similarity(id_embedding, gen_img_embedding).numpy())
    #         similarities['id_to_attr'].extend(tf.keras.losses.cosine_similarity(id_embedding, attr_img_id_embedding).numpy())
    #         similarities['attr_to_pred'].extend(tf.keras.losses.cosine_similarity(attr_img_id_embedding, gen_img_embedding).numpy())

    #         # Landmarks
    #         dst_lnds = self.model.G.landmarks(gen_img)
    #         lnd_dist.extend(tf.reduce_mean(tf.keras.losses.MSE(attr_lnds, dst_lnds), axis=-1).numpy())

    #         # Fake Reconstruction
    #         self.test_reconstruction(id_img, id_z_matching, fake_reconstruction, display=(i==0), display_name='id_img')

    #         # if self.args.test_real_attr:
    #         #     # Real Reconstruction
    #         #     self.test_reconstruction(attr_img, attr_z_matching, real_reconstruction, display=(i==0), display_name='attr_img')

    #         if i == 0:
    #             utils.save_image(gen_img[0], self.args.images_results.joinpath(f'test_prediction_{self.num_epoch}.png'))
    #             utils.save_image(id_img[0], self.args.images_results.joinpath(f'test_id_{self.num_epoch}.png'))
    #             utils.save_image(attr_img[0], self.args.images_results.joinpath(f'test_attr_{self.num_epoch}.png'))

    #             Writer.add_image('test/prediction', gen_img, step=self.num_epoch)
    #             Writer.add_image('test input/id image', id_img, step=self.num_epoch)
    #             Writer.add_image('test input/attr image', attr_img, step=self.num_epoch)

    #             for j in range(np.minimum(3, attr_lnds.shape[0])):
    #                 src_xy = attr_lnds[j]  # GT
    #                 dst_xy = dst_lnds[j]  # pred

    #                 attr_marked = utils.mark_landmarks(attr_img[j], src_xy, color=(0, 0, 0))
    #                 pred_marked = utils.mark_landmarks(gen_img[j], src_xy, color=(0, 0, 0))
    #                 pred_marked = utils.mark_landmarks(pred_marked, dst_xy, color=(255, 112, 112))

    #                 Writer.add_image(f'landmarks/overlay-{j}', pred_marked, step=self.num_epoch)
    #                 Writer.add_image(f'landmarks/src-{j}', attr_marked, step=self.num_epoch)

    #     # Similarity
    #     self.logger.info('Similarities:')
    #     for k, v in similarities.items():
    #         self.logger.info(f'{k}: MEAN: {np.mean(v)}, STD: {np.std(v)}')

    #     mean_lnd_dist = np.mean(lnd_dist)
    #     self.logger.info(f'Mean landmarks L2: {mean_lnd_dist}')

    #     id_to_pred = np.mean(similarities['id_to_pred'])
    #     attr_to_pred = np.mean(similarities['attr_to_pred'])
    #     mean_disen = attr_to_pred - id_to_pred

    #     Writer.add_scalar('similarity/score', mean_disen, step=self.num_epoch)
    #     Writer.add_scalar('similarity/id_to_pred', id_to_pred, step=self.num_epoch)
    #     Writer.add_scalar('similarity/attr_to_pred', attr_to_pred, step=self.num_epoch)

    #     if self.args.test_with_arcface:
    #         arc_id_to_pred = np.mean(test_similarities['id_to_pred'])
    #         arc_attr_to_pred = np.mean(test_similarities['attr_to_pred'])
    #         arc_mean_disen = arc_attr_to_pred - arc_id_to_pred

    #         Writer.add_scalar('arc_similarity/score', arc_mean_disen, step=self.num_epoch)
    #         Writer.add_scalar('arc_similarity/id_to_pred', arc_id_to_pred, step=self.num_epoch)
    #         Writer.add_scalar('arc_similarity/attr_to_pred', arc_attr_to_pred, step=self.num_epoch)

    #     self.logger.info(f'Mean disentanglement score is {mean_disen}')

    #     Writer.add_scalar('landmarks/L2', np.mean(lnd_dist), step=self.num_epoch)

    #     # Reconstruction
    #     # if self.args.test_real_attr:
    #     #     Writer.add_scalar('reconstruction/real_MSE', np.mean(real_reconstruction['MSE']), step=self.num_epoch)
    #     #     Writer.add_scalar('reconstruction/real_PSNR', np.mean(real_reconstruction['PSNR']), step=self.num_epoch)
    #     #     Writer.add_scalar('reconstruction/real_ID', np.mean(real_reconstruction['ID']), step=self.num_epoch)

    #     Writer.add_scalar('reconstruction/fake_MSE', np.mean(fake_reconstruction['MSE']), step=self.num_epoch)
    #     Writer.add_scalar('reconstruction/fake_PSNR', np.mean(fake_reconstruction['PSNR']), step=self.num_epoch)
    #     Writer.add_scalar('reconstruction/fake_ID', np.mean(fake_reconstruction['ID']), step=self.num_epoch)

    #     # if mean_lnd_dist < self.min_lnd_dist:
    #     #     self.logger.info('Minimum landmarks dist achieved. saving checkpoint')
    #     #     self.test_not_imporved = 0
    #     #     self.min_lnd_dist = mean_lnd_dist
    #     #     self.model.my_save(f'_best_landmarks_epoch_{self.num_epoch}')

    #     # if np.abs(id_to_pred) > self.max_id_preserve:
    #     #     self.logger.info(f'Max ID preservation achieved! saving checkpoint')
    #     #     self.test_not_imporved = 0
    #     #     self.max_id_preserve = np.abs(id_to_pred)
    #     #     self.model.my_save(f'_best_id_epoch_{self.num_epoch}')

    #     # else:
    #     #     self.test_not_imporved += 1

    # def test_reconstruction(self, img, z_matching, errors_dict, display=False, display_name=None):

    #     gen_img, id_embedding, _, _ = self.model.G(img, z_matching, img, self.ref_origin)

    #     recon_image = tf.clip_by_value(gen_img, 0, 1)
    #     recon_pred_id = self.model.G.id_encoder(recon_image)

    #     mse = tf.reduce_mean((img - recon_image) ** 2, axis=[1, 2, 3]).numpy()
    #     psnr = tf.image.psnr(img, recon_image, 1).numpy()

    #     errors_dict['MSE'].extend(mse)
    #     errors_dict['PSNR'].extend(psnr)
    #     errors_dict['ID'].extend(tf.keras.losses.cosine_similarity(id_embedding, recon_pred_id).numpy())

    #     if display:
    #         Writer.add_image(f'reconstruction/input_{display_name}_img', img, step=self.num_epoch)
    #         Writer.add_image(f'reconstruction/pred_{display_name}_img', gen_img, step=self.num_epoch)

    # Helpers
    # def generator_gan_loss(self, fake_logit):
    #     """
    #     G logistic non saturating loss, to be minimized
    #     """
    #     g_gan_loss = self.gan_loss_func(tf.ones_like(fake_logit), fake_logit)
    #     return self.lambda_gan * g_gan_loss

    # def discriminator_loss(self, fake_logit, real_logit):
    #     """
    #     D logistic loss, to be minimized
    #     verified as identical to StyleGAN' loss.D_logistic
    #     """
    #     fake_gt = tf.zeros_like(fake_logit)
    #     real_gt = tf.ones_like(real_logit)

    #     d_fake_loss = self.gan_loss_func(fake_gt, fake_logit)
    #     d_real_loss = self.gan_loss_func(real_gt, real_logit)

    #     d_loss = d_real_loss + d_fake_loss

    #     return self.lambda_gan * d_loss

    # def R1_gp(self, D, x):
    #     with tf.GradientTape() as t:
    #         t.watch(x)
    #         pred = D(x)
    #         pred_sum = tf.reduce_sum(pred)

    #     grad = t.gradient(pred_sum, x)

    #     # Reshape as a vector
    #     norm = tf.norm(tf.reshape(grad, [tf.shape(grad)[0], -1]), axis=1)
    #     gp = tf.reduce_mean(norm ** 2)
    #     gp = 0.5 * self.r1_gamma * gp

    #     return gp
    
    def get_id_attr_sampler(self, dataset_length, cross_frequency):
        split = dataset_length // (cross_frequency + 1)
        id_dataset_length = dataset_length - split
        indices = list(range(dataset_length))
        random.shuffle(indices)
        id_sampler =  SubsetRandomSampler(indices[split:])
        attr_sampler = SubsetRandomSampler(indices[:split])
        return id_sampler, attr_sampler, id_dataset_length