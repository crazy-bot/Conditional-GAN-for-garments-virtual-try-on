# Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

import numpy as np
import torch
import torch.nn as nn
from torch_utils import training_stats
from torch_utils import misc
from torch_utils.ops import conv2d_gradfix
from training.vgg16 import Vgg16
from metrics import metric_utils
#----------------------------------------------------------------------------
vgg_model = Vgg16().cuda()


class Loss:
    
    def accumulate_gradients(self, phase, real_img, gen_z, gen_pose, sync, gain): # to be overridden by subclass
        raise NotImplementedError()

    # losses

#----------------------------------------------------------------------------

class StyleGAN2Loss(Loss):
    def __init__(self, device, G, D, num_gpus=1, rank=0, augment_pipe=None, style_mixing_prob=0.9, r1_gamma=10, pl_batch_shrink=2, pl_decay=0.01, pl_weight=0):
        super().__init__()
        self.device = device
        self.G = G
        self.D = D
        self.augment_pipe = augment_pipe
        self.style_mixing_prob = style_mixing_prob
        self.r1_gamma = r1_gamma
        self.pl_batch_shrink = pl_batch_shrink
        self.pl_decay = pl_decay
        self.pl_weight = pl_weight
        self.pl_mean = torch.zeros([], device=device)
        vgg16_url = 'https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/metrics/vgg16.pt'
        vgg_model = metric_utils.get_feature_detector(vgg16_url, device=device, num_gpus=num_gpus, rank=rank)
        self.vgg_model = vgg_model
        

    def perceptual_loss(self, img1, img2, sync):
        with misc.ddp_sync(self.vgg_model, sync):
            weight = [1.0 / 32, 1.0 / 16, 1.0 / 8, 1.0 / 4, 1.0]  # TODO: to be verified
            weight.reverse()
            img1_features = self.vgg_model(img1.float())
            img2_features = self.vgg_model(img2.float())
            loss = nn.L1Loss().cuda()
            weighted_dist = 0
            for i in range(4):
                weighted_dist = weighted_dist + weight[i] * loss(img1_features[i], img2_features[i])
        return weighted_dist


    def l1_loss(self,image1, image2):
        loss = nn.L1Loss()
        return loss(image1, image2)

    # def run_G(self, z, pose, sync):
    #     with misc.ddp_sync(self.G_mapping, sync):
    #         ws = self.G_mapping(z)
    #         if self.style_mixing_prob > 0:
    #             with torch.autograd.profiler.record_function('style_mixing'):
    #                 cutoff = torch.empty([], dtype=torch.int64, device=ws.device).random_(1, ws.shape[1])
    #                 cutoff = torch.where(torch.rand([], device=ws.device) < self.style_mixing_prob, cutoff, torch.full_like(cutoff, ws.shape[1]))
    #                 ws[:, cutoff:] = self.G_mapping(torch.randn_like(z), skip_w_avg_update=True)[:, cutoff:]
    #     with misc.ddp_sync(self.G_synthesis, sync):
    #         encoded_pose = self.G_encoder(pose)
    #         img = self.G_synthesis(ws, encoded_pose)
    #     return img, ws

    def run_G(self, pose, cloth, sync):
        with misc.ddp_sync(self.G, sync):
            img, ws = self.G(pose, cloth)
        return img, ws

    def run_D(self, img, sync):
        if self.augment_pipe is not None:
            img = self.augment_pipe(img)
        with misc.ddp_sync(self.D, sync):
            logits = self.D(img)
        return logits

    def accumulate_gradients(self, phase, real_img, real_pose, real_cloth, sync, gain):
        assert phase in ['Gmain', 'Greg', 'Gboth', 'Dmain', 'Dreg', 'Dboth']
        do_Gmain = (phase in ['Gmain', 'Gboth'])
        do_Dmain = (phase in ['Dmain', 'Dboth'])
        do_Gpl   = (phase in ['Greg', 'Gboth']) and (self.pl_weight != 0)
        do_Dr1   = (phase in ['Dreg', 'Dboth']) and (self.r1_gamma != 0)

        # Gmain: Maximize logits for generated images.
        if do_Gmain:
            with torch.autograd.profiler.record_function('Gmain_forward'):
                gen_img, _ = self.run_G(real_pose, real_cloth, sync=(sync and not do_Gpl)) # May get synced by Gpl.
                gen_logits = self.run_D(gen_img, sync=False)

                training_stats.report('Loss/scores/fake', gen_logits)
                #training_stats.report('Loss/signs/fake', gen_logits.sign())

                loss_Gmain = torch.nn.functional.softplus(-gen_logits) # -log(sigmoid(gen_logits))

                loss_rec = self.l1_loss(real_img, gen_img) + self.perceptual_loss(real_img, gen_img, sync) + loss_Gmain
                #loss_rec = loss_Gmain + self.perceptual_loss(real_img, gen_img) +
                training_stats.report('Loss/G/loss', loss_rec)

            with torch.autograd.profiler.record_function('Gmain_backward'):
                loss_rec.mean().mul(gain).backward()

        # Gpl: Apply path length regularization.
        if do_Gpl:
            with torch.autograd.profiler.record_function('Gpl_forward'):
                batch_size = real_pose.shape[0] // self.pl_batch_shrink
                gen_img, gen_ws = self.run_G(real_pose[:batch_size], real_cloth[:batch_size], sync=sync)
                pl_noise = torch.randn_like(gen_img) / np.sqrt(gen_img.shape[2] * gen_img.shape[3])

                with torch.autograd.profiler.record_function('pl_grads'), conv2d_gradfix.no_weight_gradients():
                    pl_grads = torch.autograd.grad(outputs=[(gen_img * pl_noise).sum()], inputs=[gen_ws], create_graph=True, only_inputs=True)[0]
                pl_lengths = pl_grads.square().sum(2).mean(1).sqrt()
                pl_mean = self.pl_mean.lerp(pl_lengths.mean(), self.pl_decay)
                self.pl_mean.copy_(pl_mean.detach())
                pl_penalty = (pl_lengths - pl_mean).square()
                training_stats.report('Loss/pl_penalty', pl_penalty)
                loss_Gpl = pl_penalty * self.pl_weight
                training_stats.report('Loss/G/reg', loss_Gpl)

            with torch.autograd.profiler.record_function('Gpl_backward'):
                (gen_img[:, 0, 0, 0] * 0 + loss_Gpl).mean().mul(gain).backward()

        # Dmain: Minimize logits for generated images.
        loss_Dgen = 0
        if do_Dmain:
            with torch.autograd.profiler.record_function('Dgen_forward'):
                gen_img, _gen_ws = self.run_G(real_pose, real_cloth, sync=False)
                gen_logits = self.run_D(gen_img, sync=False) # Gets synced by loss_Dreal.
                training_stats.report('Loss/scores/fake', gen_logits)
                #training_stats.report('Loss/signs/fake', gen_logits.sign())
                loss_Dgen = torch.nn.functional.softplus(gen_logits) # -log(1 - sigmoid(gen_logits))

                #loss_rec = -(self.l1_loss(real_img, gen_img) ) + loss_Dgen
                #loss_rec = -(self.l1_loss(real_img, gen_img) + self.perceptual_loss(real_img, gen_img)) + loss_Dgen

            with torch.autograd.profiler.record_function('Dgen_backward'):
                loss_Dgen.mean().mul(gain).backward()

        # Dmain: Maximize logits for real images.
        # Dr1: Apply R1 regularization.
        if do_Dmain or do_Dr1:
            name = 'Dreal_Dr1' if do_Dmain and do_Dr1 else 'Dreal' if do_Dmain else 'Dr1'
            with torch.autograd.profiler.record_function(name + '_forward'):
                real_img_tmp = real_img.detach().requires_grad_(do_Dr1)
                real_logits = self.run_D(real_img_tmp, sync=sync)
                training_stats.report('Loss/scores/real', real_logits)
                #training_stats.report('Loss/signs/real', real_logits.sign())

                loss_Dreal = 0
                if do_Dmain:
                    loss_Dreal = torch.nn.functional.softplus(-real_logits) # -log(sigmoid(real_logits))
                    training_stats.report('Loss/D/loss',  loss_Dreal )

                loss_Dr1 = 0
                if do_Dr1:
                    with torch.autograd.profiler.record_function('r1_grads'), conv2d_gradfix.no_weight_gradients():
                        r1_grads = torch.autograd.grad(outputs=[real_logits.sum()], inputs=[real_img_tmp], create_graph=True, only_inputs=True)[0]
                    r1_penalty = r1_grads.square().sum([1,2,3])
                    loss_Dr1 = r1_penalty * (self.r1_gamma / 2)
                    training_stats.report('Loss/r1_penalty', r1_penalty)
                    training_stats.report('Loss/D/reg', loss_Dr1)

            with torch.autograd.profiler.record_function(name + '_backward'):
                (real_logits * 0 + loss_Dreal + loss_Dr1).mean().mul(gain).backward()

#----------------------------------------------------------------------------
