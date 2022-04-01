import os
import argparse

import yaml
import numpy as np
from PIL import Image

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from libs.config import load_config
from libs.data import collate_fn, make_dataset
from libs.encoder import make_rsd
from libs.metrics import RMSE, PSNR, SSIM
from libs.utils import *


def main(args):
    # set up save folder
    save_path = os.path.join('./rsd', args.name)
    ensure_path(save_path)

    # load config file
    check_file(args.config)
    config = load_config(args.config, mode='rsd')

    yaml.dump(config, open(os.path.join(save_path, 'config.yaml'), 'w'))

    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True
    # torch.backends.cudnn.deterministic = True

    # configure GPUs
    set_gpu(args.gpu)
    n_gpus = torch.cuda.device_count()
    if n_gpus > 1:
        config['_parallel'] = True

    ############################################################################
    """ dataset """

    dataset = make_dataset(config['dataset'], split=config['split'])
    print('dataset size: {:d}'.format(len(dataset)))

    loader = DataLoader(
        dataset, 
        batch_size=config['batch_size'],
        num_workers=config['n_workers'],
        collate_fn=collate_fn,
        shuffle=False, 
        pin_memory=True, 
        drop_last=False,
    )

    ############################################################################
    """ Phasor field RSD reconstruction """
    
    rsd = make_rsd(config['rsd'], efficient=True)
    if config.get('_parallel'):
        rsd = nn.DataParallel(rsd)
    rsd.eval()

    rmse = RMSE().cuda()
    psnr = PSNR().cuda()
    ssim = SSIM().cuda()

    rmse_metric = AverageMeter()
    psnr_metric = AverageMeter()
    ssim_metric = AverageMeter()

    idx = 1
    for (meas, target, _) in loader:
        meas = meas.cuda(non_blocking=True)
        if target is not None:
            target = target[:, 0]
            y = target.cuda(non_blocking=True)

        with torch.no_grad():
            x = rsd(meas)                       # (b, 1/3, d, h, w)

        # max projection along depth axis
        x = x.amax(dim=2)                       # (b, 1/3, h, w)
        
        x /= x.amax(dim=(1, 2, 3), keepdim=True)
        pred = x.cpu().numpy()
        pred = pred.transpose(0, 2, 3, 1)       # (b, h, w, 1/3)

        if target is not None:
            target = target.numpy().transpose(0, 2, 3, 1)
            pred = np.concatenate([target, pred], axis=2)

            rmse_metric.update(rmse(x, y).item())
            psnr_metric.update(psnr(x, y).item())
            ssim_metric.update(ssim(x, y).item())
        
        pred = (pred * 255).astype(np.uint8)
        if pred.shape[-1] == 1:
            pred = pred[..., 0]

        for i in range(len(pred)):
            im = Image.fromarray(pred[i])
            im.save(os.path.join(save_path, '{:04d}.png'.format(idx)))
            idx += 1

    print('RMSE: {:.3f}'.format(rmse_metric.item()))
    print('PSNR: {:.3f}'.format(psnr_metric.item()))
    print('SSIM: {:.3f}'.format(ssim_metric.item()))

################################################################################
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', help='config file')
    parser.add_argument('-n', '--name', help='job name')
    parser.add_argument('-g', '--gpu', help='GPU device IDs', 
                        type=str, default='0')
    args = parser.parse_args()

    main(args)