import random

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from .model import *
from .metrics import RMSE, PSNR, SSIM
from .camera import make_wall, make_camera, sample_views, get_all_views
from .loss import *


class WorkerBase:

    def __init__(self):

        self.model = None
        self.n_gpus = 1

        self.rmse = RMSE()
        self.psnr = PSNR()
        self.ssim = SSIM()

    def cuda(self, n_gpus=1):
        self.model.cuda()
        self.rmse.cuda()
        self.psnr.cuda()
        self.ssim.cuda()

        self.n_gpus = n_gpus
        if n_gpus > 1:
            self.model = nn.DataParallel(self.model)

    def load(self, ckpt):
        if self.n_gpus > 1:
            self.model.module.load_state_dict(ckpt['model'])
        else:
            self.model.load_state_dict(ckpt['model'])

    def save(self):
        if self.n_gpus > 1:
            model = self.model.module.state_dict()
        else:
            model = self.model.state_dict()
        ckpt = {'model': model}
        return ckpt

    def parameters(self):
        if self.n_gpus > 1:
            return self.model.module.parameters()
        else:
            return self.model.parameters()


class RendererWorkerBase(WorkerBase):

    def __init__(self, cam_cfg, model_cfg):
        super(RendererWorkerBase, self).__init__()

        self.cam = make_camera(cam_cfg)
        self.model = make_renderer_model(model_cfg)

        # buffer all rays
        Rt_all = get_all_views()
        rays = []
        for idx in range(len(Rt_all)):
            r = self.cam.get_all_rays(Rt_all[idx], invert_z=True)
            r = torch.from_numpy(r.astype(np.float32))
            rays.append(r)
        self.rays = torch.stack(rays)   # (v, h, w, 7)

    @torch.no_grad()
    def eval(self, target, cfg):
        """
        Args:
            target (float tensor, (v, 1/3, h, w)): target images.
            cfg (dict): evaluation config.
        """
        # sample target views
        if cfg['n_views'] > 0:
            view_idx, Rt = sample_views(
                n_views=cfg['n_views'],
                include_orthogonal=cfg['include_orthogonal'],
            )
        else:
            Rt = get_all_views()
            view_idx = list(range(len(Rt)))

        target = target[view_idx]
        v, _, h, w = target.size()
        target = target.cuda(non_blocking=True)

        # get all rays originaing from sampled views
        rays = self.rays[view_idx]
        rays = rays.flatten(1, 2)                       # (v, h*w, 7)
        assert h * w % self.n_gpus == 0
        rays = rays.reshape(v, self.n_gpus, h * w // self.n_gpus, 7)
        rays = rays.transpose(0, 1)                     # (g, v, h*w/g, 7)
        rays = rays.cuda(non_blocking=True)
        
        # batchify rays
        chunk_size = cfg['chunk_size'] // len(view_idx)
        chunks = [chunk_size] * (rays.size(-2) // chunk_size)
        chunks[-1] += rays.size(-2) % chunk_size
        batched_rays = rays.split(chunks, dim=-2)

        self.model.eval()
        lib = dict()
        for r in batched_rays:
            _, lib_batch = self.model(
                wall_rays=None,
                cam_rays=r,
                n_steps=cfg['n_steps'],
                s_scale=cfg.get('s_scale', 1),
                sigma_noise=cfg.get('sigma_noise', 0),
                color_noise=cfg.get('color_noise', 0),
            )
            for k in lib_batch.keys():
                if k not in lib:
                    lib[k] = lib_batch[k]
                else:
                    lib[k] = torch.cat([lib[k], lib_batch[k]], dim=2)

        pred = lib['render'].reshape(v, h, w, -1)       # (v, h, w, 1/3)
        pred = pred.permute(0, 3, 1, 2)                 # (v, 1/3, h, w)

        if cfg.get('normalize'):
            pred /= pred.amax(dim=(-3, -2, -1), keepdim=True) + 1e-8
            target /= target.amax(dim=(-3, -2, -1), keepdim=True) + 1e-8

        pred = torch.clamp(pred, 0, 1)
        target = torch.clamp(target, 0, 1)

        loss = F.mse_loss(pred, target, reduction='mean')

        output_dict = {
            'pred': pred.cpu(),
            'target': target.cpu(),
        }

        metric_dict = {
            'rmse': self.rmse(pred, target).cpu(),
            'psnr': self.psnr(pred, target).cpu(),
            'ssim': self.ssim(pred, target).cpu(),
        }

        return loss, output_dict, metric_dict


class UnsupRendererWorker(RendererWorkerBase):

    def __init__(self, wall_cfg, cam_cfg, model_cfg):
        super(UnsupRendererWorker, self).__init__(cam_cfg, model_cfg)

        self.wall = make_wall(wall_cfg)

    def train(self, meas, cfg):
        """
        Args:
            meas (float tensor, (1/3, t, h, w)): measurement.
            cfg (dict): training config.
        """
        meas = meas.cuda(non_blocking=True)

        # sample rays originating from the wall
        spad_idx, rays = self.wall.sample_rays(invert_z=True)
        spad_idx = torch.from_numpy(spad_idx[0]).long() # (v, 2)
        rays = torch.from_numpy(rays.astype(np.float32))# (1, v, n, 7)
        v = rays.size(1)
        assert v % self.n_gpus == 0
        rays = rays.reshape(self.n_gpus, v // self.n_gpus, -1, 7)
        rays = rays.cuda(non_blocking=True)

        self.model.train()
        lib, _ = self.model(
            wall_rays=rays, 
            cam_rays=None,
            n_steps=cfg['n_steps'],
            t_scale=cfg.get('t_scale', 1),
            sigma_noise=cfg.get('sigma_noise', 0),
            color_noise=cfg.get('color_noise', 0),
        )

        for k in lib.keys():
            lib[k] = lib[k].flatten(0, 1)

        # fetch target histograms
        hists = meas.permute(2, 3, 1, 0)               # (h, w, t, 1/3)
        b0, b1 = cfg['bin_range']
        hists = hists[spad_idx[:, 0], spad_idx[:, 1], b0:b1]

        p_loss = poisson_nll_loss(lib['render'], hists, reduction='mean')
        b_loss = beta_loss(lib['hit'], log_space=True, reduction='mean')
        t_loss = tv_loss(lib['alpha'], log_space=True, reduction='mean')
        
        loss_dict = {
            'poisson': p_loss,
            'beta': b_loss,
            'tv': t_loss,
        }

        pred = lib['render'].detach()                   # (v, t, 1/3)
        pred = pred.transpose(-2, -1)                   # (v, 1/3, t)
        target = hists.transpose(-2, -1)                # (v, 1/3, t)

        output_dict = {
            'pred': pred.cpu(),
            'target': target.cpu(),
        }

        return loss_dict, output_dict


class SupRendererWorker(RendererWorkerBase):

    def __init__(self, cam_cfg, model_cfg):
        super(SupRendererWorker, self).__init__(cam_cfg, model_cfg)

    def train(self, target, cfg):
        """
        Args:
            target (float tensor, (v, 1/3, h, w)): target images.
            cfg (dict): training config.
        """
        # sample rays originating from camera views
        if cfg.get('n_rays', -1) > 0:
            rays = self.rays.flatten(0, 2)                  # (k, 7)
            k = len(rays)
            assert cfg['n_rays'] < k
            idx = random.sample(range(k), cfg['n_rays'])
            rays = rays[None, idx]                          # (1, n, 7)
            rays = rays.reshape(self.n_gpus, 1, -1, 7)      # (g, 1, n/g, 7)
            pixels = target.permute(0, 2, 3, 1)             # (v, h, w, 1/3)
            pixels = pixels.flatten(0, 2)
            pixels = pixels[None, idx]                      # (1, n, 1/3)
        else:
            # sample target views
            view_idx, Rt = sample_views(
                n_views=cfg['n_views'],
                include_orthogonal=cfg['include_orthogonal'],
            )
            target = target[view_idx]
            v = target.size(0)
            
            # sample rays originating from target views
            pixels, rays = [], []
            for idx in range(len(view_idx)):
                px_idx, r = self.cam.sample_rays(Rt[idx], invert_z=True)
                px_idx = torch.from_numpy(px_idx[0]).long() # (n, 2)
                px = target[idx, :, px_idx[:, 0], px_idx[:, 1]]
                px = px.transpose(0, 1)                     # (r, 1/3)
                pixels.append(px)
                r = torch.from_numpy(r.astype(np.float32))  # (1, r, 7)
                n = r.size(1)
                assert n % self.n_gpus == 0
                r = r.reshape(self.n_gpus, n // self.n_gpus, 7)
                rays.append(r)
            pixels = torch.stack(pixels)                    # (v, r, 1/3)
            rays = torch.stack(rays, dim=1)                 # (g, v, r/g, 7)

        pixels = pixels.cuda(non_blocking=True)
        rays = rays.cuda(non_blocking=True)

        self.model.train()
        _, lib = self.model(
            wall_rays=None,
            cam_rays=rays,
            n_steps=cfg['n_steps'],
            s_scale=cfg.get('s_scale', 1),
            sigma_noise=cfg.get('sigma_noise', 0),
            color_noise=cfg.get('color_noise', 0),
        )

        for k in lib.keys():
            lib[k] = lib[k].transpose(0, 1).flatten(1, 2)
        
        m_loss = F.mse_loss(lib['render'], pixels, reduction='mean')
        b_loss = beta_loss(lib['hit'], log_space=True, reduction='mean')
        t_loss = tv_loss(lib['alpha'], log_space=True, reduction='mean')

        loss_dict = {
            'mse': m_loss,
            'beta': b_loss,
            'tv': t_loss,
        }

        pred = lib['render'].detach()
        pred = pred.flatten(0, 1).transpose(0, 1)       # (1/3, v*r (or n))
        target = pixels.flatten(0, 1).transpose(0, 1)   # (1/3, v*r (or n))

        output_dict = {
            'pred': pred.cpu(),
            'target': target.cpu(),
        }

        return loss_dict, output_dict


class EncoderRendererWorkerBase(WorkerBase):

    def __init__(self, cam_cfg, model_cfg):
        super(EncoderRendererWorkerBase, self).__init__()

        self.cam = make_camera(cam_cfg)
        self.model = make_encoder_renderer_model(model_cfg)

        # buffer all rays
        Rt_all = get_all_views()
        rays = []
        for idx in range(len(Rt_all)):
            r = self.cam.get_all_rays(Rt_all[idx], invert_z=True)
            r = torch.from_numpy(r.astype(np.float32))
            rays.append(r)
        self.rays = torch.stack(rays)   # (v, h, w, 7)

    @torch.no_grad()
    def eval(self, meas, target, cfg):
        """
        Args:
            meas (float tensor, (bs, 1/3, t, h, w)): measurements.
            target (float tensor, (bs, v, 1/3, h, w)): target images.
            cfg (dict): evaluation config.
        """
        meas = meas.cuda(non_blocking=True)

        # sample target views
        if cfg['n_views'] > 0:
            view_idx, Rt = sample_views(
                n_views=cfg['n_views'],
                include_orthogonal=cfg['include_orthogonal'],
            )
        else:
            Rt = get_all_views()
            view_idx = list(range(len(Rt)))

        target = target[:, view_idx]
        bs, v, _, h, w = target.size()
        target = target.cuda(non_blocking=True)

        # get all rays originaing from sampled views
        rays = self.rays[view_idx]
        rays = rays.flatten(1, 2)                       # (v, h*w, 7)
        rays = rays.repeat(self.n_gpus, 1, 1, 1)        # (g, v, h*w, 7)
        rays = rays.cuda(non_blocking=True)
        
        # batchify rays
        chunk_size = cfg['chunk_size'] // len(view_idx)
        chunks = [chunk_size] * (rays.size(-2) // chunk_size)
        chunks[-1] += rays.size(-2) % chunk_size
        batched_rays = rays.split(chunks, dim=-2)

        self.model.eval()
        lib = dict()
        for r in batched_rays:
            _, lib_batch, _ = self.model(
                meas=meas,
                wall_rays=None,
                cam_rays=r,
                n_steps=cfg['n_steps'],
                in_scale=cfg.get('in_scale', 1),
                s_scale=cfg.get('s_scale', 1),
                sigma_noise=cfg.get('sigma_noise', 0),
                color_noise=cfg.get('color_noise', 0),
            )
            for k in lib_batch.keys():
                if k not in lib:
                    lib[k] = lib_batch[k]
                else:
                    lib[k] = torch.cat([lib[k], lib_batch[k]], dim=2)

        pred = lib['render'].reshape(bs, v, h, w, -1)   # (bs, v, h, w, 1/3)
        pred = pred.permute(0, 1, 4, 2, 3)              # (bs, v, 1/3, h, w)

        if cfg.get('normalize'):
            pred /= pred.amax(dim=(-3, -2, -1), keepdim=True) + 1e-8
            target /= target.amax(dim=(-3, -2, -1), keepdim=True) + 1e-8

        pred = torch.clamp(pred, 0, 1)
        target = torch.clamp(target, 0, 1)
        pred = pred.flatten(0, 1)                       # (bs*v, h, w, 1/3)
        target = target.flatten(0, 1)                   # (bs*v, h, w, 1/3)

        loss = F.mse_loss(pred, target, reduction='mean')

        output_dict = {
            'pred': pred.cpu(),
            'target': target.cpu(),
        }

        metric_dict = {
            'rmse': self.rmse(pred, target).cpu(),
            'psnr': self.psnr(pred, target).cpu(),
            'ssim': self.ssim(pred, target).cpu(),
        }

        return loss, output_dict, metric_dict


class UnsupEncoderRendererWorker(EncoderRendererWorkerBase):

    def __init__(self, wall_cfg, cam_cfg, model_cfg):
        super(UnsupEncoderRendererWorker, self).__init__(cam_cfg, model_cfg)

        self.wall = make_wall(wall_cfg)

    def train(self, meas, cfg):
        """
        Args:
            meas (float tensor, (bs, 1/3, t, h, w)): measurements.
            cfg (dict): training config.
        """
        meas = meas.cuda(non_blocking=True)

        # sample rays originating from the wall
        spad_idx, rays = self.wall.sample_rays(invert_z=True)
        spad_idx = torch.from_numpy(spad_idx[0]).long() # (v, 2)
        rays = torch.from_numpy(rays.astype(np.float32))# (1, v, n, 7)
        rays = rays.repeat(self.n_gpus, 1, 1, 1)        # (g, v, n, 7)
        rays = rays.cuda(non_blocking=True)

        self.model.train()
        lib, _, _ = self.model(
            meas=meas,
            wall_rays=rays,
            cam_rays=None,
            n_steps=cfg['n_steps'],
            in_scale=cfg.get('in_scale', 1),
            t_scale=cfg.get('t_scale', 1),
            sigma_noise=cfg.get('sigma_noise', 0),
            color_noise=cfg.get('color_noise', 0),
        )

        # fetch target histograms
        hists = meas.permute(0, 3, 4, 2, 1)             # (bs, h, w, t, 1/3)
        b0, b1 = cfg['bin_range']
        hists = hists[:, spad_idx[:, 0], spad_idx[:, 1], b0:b1]

        p_loss = poisson_nll_loss(lib['render'], hists, reduction='mean')
        b_loss = beta_loss(lib['hit'], log_space=True, reduction='mean')
        t_loss = tv_loss(lib['alpha'], log_space=True, reduction='mean')
        
        loss_dict = {
            'poisson': p_loss,
            'beta': b_loss,
            'tv': t_loss,
        }

        pred = lib['render'].detach()
        pred = pred.transpose(-2, -1).flatten(0, 1)     # (bs*v, 1/3, t)
        target = hists.transpose(-2, -1).flatten(0, 1)  # (bs*v, 1/3, t)

        output_dict = {
            'pred': pred.cpu(),
            'target': target.cpu(),
        }

        return loss_dict, output_dict


class SupEncoderRendererWorker(EncoderRendererWorkerBase):

    def __init__(self, cam_cfg, model_cfg):
        super(SupEncoderRendererWorker, self).__init__(cam_cfg, model_cfg)

    def train(self, meas, target, cfg):
        """
        Args:
            meas (float tensor, (bs, 1/3, t, h, w)): measurements.
            target (float tensor, (bs, v, 1/3, h, w)): target images.
            cfg (dict): training config.
        """
        meas = meas.cuda(non_blocking=True)

        # sample rays originating from camera views
        if cfg.get('n_rays', -1) > 0:
            rays = self.rays.flatten(0, 2)                  # (k, 7)
            k = len(rays)
            assert cfg['n_rays'] < k
            idx = random.sample(range(k), cfg['n_rays'])
            rays = rays[None, idx]                          # (1, n, 7)
            rays = rays.repeat(self.n_gpus, 1, 1, 1)        # (g, 1, n, 7)
            pixels = target.permute(0, 1, 3, 4, 2)          # (bs, v, h, w, 1/3)
            pixels = pixels.flatten(1, 3)
            pixels = pixels[:, None, idx]                   # (bs, 1, n, 1/3)
        else:
            # sample target views
            view_idx, Rt = sample_views(
                n_views=cfg['n_views'],
                include_orthogonal=cfg['include_orthogonal'],
            )
            target = target[:, view_idx]
            bs, v = target.shape[:2]
            
            # sample rays originating from target views
            pixels, rays = [], []
            for idx in range(len(view_idx)):
                px_idx, r = self.cam.sample_rays(Rt[idx], invert_z=True)
                px_idx = torch.from_numpy(px_idx[0]).long() # (n, 2)
                px = target[:, idx, :, px_idx[:, 0], px_idx[:, 1]]
                px = px.transpose(1, 2)                     # (bs, r, 1/3)
                pixels.append(px)
                r = torch.from_numpy(r.astype(np.float32))  # (1, r, 7)
                r = r.repeat(self.n_gpus, 1, 1)             # (g, r, 7)
                rays.append(r)
            pixels = torch.stack(pixels, dim=1)             # (bs, v, r, 1/3)
            rays = torch.stack(rays, dim=1)                 # (g, v, r, 7)
        
        pixels = pixels.cuda(non_blocking=True)
        rays = rays.cuda(non_blocking=True)

        self.model.train()
        _, lib, _ = self.model(
            meas=meas,
            wall_rays=None,
            cam_rays=rays,
            n_steps=cfg['n_steps'],
            in_scale=cfg.get('in_scale', 1),
            s_scale=cfg.get('s_scale', 1),
            sigma_noise=cfg.get('sigma_noise', 0),
            color_noise=cfg.get('color_noise', 0),
        )
        
        m_loss = F.mse_loss(lib['render'], pixels, reduction='mean')
        b_loss = beta_loss(lib['hit'], log_space=True, reduction='mean')
        t_loss = tv_loss(lib['alpha'], log_space=True, reduction='mean')

        loss_dict = {
            'mse': m_loss,
            'beta': b_loss,
            'tv': t_loss,
        }

        pred = lib['render'].detach()
        pred = pred.flatten(1, 2).transpose(1, 2)       # (bs, 1/3, v*r (or n))
        target = pixels.flatten(1, 2).transpose(1, 2)   # (bs, 1/3, v*r (or n))

        output_dict = {
            'pred': pred.cpu(),
            'target': target.cpu(),
        }

        return loss_dict, output_dict


class JointEncoderRendererWorker(EncoderRendererWorkerBase):

    def __init__(self, wall_cfg, cam_cfg, model_cfg):
        super(JointEncoderRendererWorker, self).__init__(cam_cfg, model_cfg)

        self.wall = make_wall(wall_cfg)

    def train(self, meas, target, cfg):
        """
        Args:
            meas (float tensor, (bs, 1/3, t, h, w)): measurements.
            target (float tensor, (bs, v, 1/3, h, w)): target images.
            cfg (dict): training config.
        """
        meas = meas.cuda(non_blocking=True)

        # sample rays originating from the wall
        spad_idx, wall_rays = self.wall.sample_rays(invert_z=True)
        spad_idx = torch.from_numpy(spad_idx[0]).long()     # (v, 2)
        wall_rays = torch.from_numpy(wall_rays.astype(np.float32))
        wall_rays = wall_rays.repeat(self.n_gpus, 1, 1, 1)
        wall_rays = wall_rays.cuda(non_blocking=True)       # (g, v, n, 7)
        
        # sample rays originating from camera views
        if cfg.get('n_rays', -1) > 0:
            cam_rays = self.rays.flatten(0, 2)              # (k, 7)
            k = len(cam_rays)
            assert cfg['n_rays'] < k
            idx = random.sample(range(k), cfg['n_rays'])
            cam_rays = cam_rays[None, idx]                  # (1, n, 7)
            cam_rays = cam_rays.repeat(self.n_gpus, 1, 1, 1)# (g, 1, n, 7)
            pixels = target.permute(0, 1, 3, 4, 2)          # (bs, v, h, w, 1/3)
            pixels = pixels.flatten(1, 3)
            pixels = pixels[:, None, idx]                   # (bs, 1, n, 1/3)
        else:
            # sample target views
            view_idx, Rt = sample_views(
                n_views=cfg['n_views'],
                include_orthogonal=cfg['include_orthogonal'],
            )
            target = target[:, view_idx]
            bs, v = target.shape[:2]

            # sample rays originating from target views
            pixels, cam_rays = [], []
            for idx in range(len(view_idx)):
                px_idx, r = self.cam.sample_rays(Rt[idx], invert_z=True)
                px_idx = torch.from_numpy(px_idx[0]).long() # (n, 2)
                px = target[:, idx, :, px_idx[:, 0], px_idx[:, 1]]
                px = px.transpose(1, 2)                     # (bs, r, 1/3)
                pixels.append(px)
                r = torch.from_numpy(r.astype(np.float32))  # (1, r, 7)
                r = r.repeat(self.n_gpus, 1, 1)             # (g, r, 7)
                cam_rays.append(r)
            pixels = torch.stack(pixels, dim=1)             # (bs, v, r, 1/3)
            cam_rays = torch.stack(cam_rays, dim=1)         # (g, v, r, 7)

        pixels = pixels.cuda(non_blocking=True)
        cam_rays = cam_rays.cuda(non_blocking=True)

        self.model.train()
        t_lib, s_lib, _ = self.model(
            meas=meas,
            wall_rays=wall_rays,
            cam_rays=cam_rays,
            n_steps=cfg['n_steps'],
            in_scale=cfg.get('in_scale', 1),
            t_scale=cfg.get('t_scale', 1),
            s_scale=cfg.get('s_scale', 1),
            sigma_noise=cfg.get('sigma_noise', 0),
            color_noise=cfg.get('color_noise', 0),
        )

        # fetch target histograms
        hists = meas.permute(0, 3, 4, 2, 1)             # (bs, h, w, t, 1/3)
        b0, b1 = cfg['bin_range']
        hists = hists[:, spad_idx[:, 0], spad_idx[:, 1], b0:b1]
        
        p_loss = poisson_nll_loss(t_lib['render'], hists, reduction='mean')
        m_loss = F.mse_loss(s_lib['render'], pixels, reduction='mean')

        n_rays = t_lib['hit'].numel() + s_lib['hit'].numel()
        b_loss = (
            beta_loss(t_lib['hit'], log_space=True, reduction='sum') + \
            beta_loss(s_lib['hit'], log_space=True, reduction='sum')
        ) / n_rays
        t_loss = (
            tv_loss(t_lib['alpha'], log_space=True, reduction='sum') + \
            tv_loss(s_lib['alpha'], log_space=True, reduction='sum')
        ) / n_rays

        loss_dict = {
            'poisson': p_loss,
            'mse': m_loss,
            'beta': b_loss,
            'tv': t_loss,
        }

        pred = t_lib['render'].detach()
        pred = pred.transpose(-2, -1).flatten(0, 1)     # (bs*v, 1/3, t)
        target = hists.transpose(-2, -1).flatten(0, 1)  # (bs*v, 1/3, t)

        t_output_dict = {
            'pred': pred.cpu(),
            'target': target.cpu(),
        }

        pred = s_lib['render'].detach()
        pred = pred.flatten(1, 2).transpose(1, 2)       # (bs, 1/3, v*r)
        target = pixels.flatten(1, 2).transpose(1, 2)   # (bs, 1/3, v*r)

        s_output_dict = {
            'pred': pred.cpu(),
            'target': target.cpu(),
        }

        return loss_dict, t_output_dict, s_output_dict


class EncoderDecoderWorker(WorkerBase):

    def __init__(self, model_cfg):
        super(EncoderDecoderWorker, self).__init__()

        self.model = make_encoder_decoder_model(model_cfg)

    def run(self, meas, target, cfg):
        """
        Args:
            meas (float tensor, (bs, 1/3, t, h, w)): measurements.
            target (float tensor, (bs, v, 1/3, h, w)): target images.
            cfg (dict): config.
        """
        meas = meas.cuda(non_blocking=True)             # (bs, 1/3, t, h, w)

        # sample target views
        if cfg['n_views'] > 0:
            view_idx, Rt = sample_views(
                n_views=cfg['n_views'],
                include_orthogonal=cfg['include_orthogonal'],
            )
        else:
            Rt = get_all_views()
            view_idx = list(range(len(Rt)))

        rot = None
        if view_idx != [0]:
            rot = Rt[:, :3, :3]
            rot = torch.from_numpy(rot.astype(np.float32))
            rot = rot.repeat(self.n_gpus, 1, 1)
            rot = rot.cuda(non_blocking=True)

        target = target[:, view_idx]                    # (bs, v, 1/3, h, w)
        target = target.cuda(non_blocking=True)

        pred, _ = self.model(meas, rot, cfg.get('in_scale', 1))

        loss = F.mse_loss(pred, target, reduction='mean')

        pred = torch.clamp(pred.detach(), 0, 1)
        target = torch.clamp(target, 0, 1)
        pred = pred.flatten(0, 1)
        target = target.flatten(0, 1)

        output_dict = {
            'pred': pred.cpu(),
            'target': target.cpu(),
        }

        metric_dict = {
            'rmse': self.rmse(pred, target).cpu(),
            'psnr': self.psnr(pred, target).cpu(),
            'ssim': self.ssim(pred, target).cpu(),
        }

        return loss, output_dict, metric_dict

    def train(self, meas, target, cfg):
        self.model.train()
        loss, output_dict, metric_dict = self.run(meas, target, cfg)
        return loss, output_dict, metric_dict

    @torch.no_grad()
    def eval(self, meas, target, cfg):
        self.model.eval()
        loss, output_dict, metric_dict = self.run(meas, target, cfg)
        return loss, output_dict, metric_dict