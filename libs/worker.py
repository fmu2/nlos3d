import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from .model import make_encoder_decoder_model
from .metrics import RMSE, PSNR, SSIM
from .camera import sample_views
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


class EncoderDecoderWorker(WorkerBase):

    def __init__(self, model_cfg):
        super(EncoderDecoderWorker, self).__init__()

        self.model = make_encoder_decoder_model(model_cfg)

    def run(self, meas, target, cfg):
        """
        Args:
            meas (float tensor, (bs, 1/3, t, h, w)): measurements.
            target (float tensor, (bs, v, h, w, 3)): target images.
        """
        meas = meas.cuda(non_blocking=True)         # (bs, 1/3, t, h, w)

        # sample target views
        view_idx, Rt = sample_views(
            n_views=cfg['n_views'],
            include_orthogonal=cfg['include_orthogonal'],
        )

        rot = None
        if view_idx != [0]:
            rot = Rt[:, :3, :3]
            rot = torch.from_numpy(rot.astype(np.float32))
            rot = rot.repeat(self.n_gpus, 1, 1)
            rot = rot.cuda(non_blocking=True)

        target = target[:, view_idx]                # (bs, v, h, w, 3)
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