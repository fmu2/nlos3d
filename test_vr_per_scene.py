import os
import argparse

import yaml
import numpy as np
from PIL import Image

import torch

from libs.config import load_config
from libs.data import make_images
from libs.worker import RendererWorkerBase
from libs.utils import *


def main(config):
    
    # fetch checkpoint folder
    ckpt_path = os.path.join('ckpt', args.ckpt)
    check_path(ckpt_path)

    # load config
    check_file(args.config)
    config = load_config(args.config, mode='test_vr_per_scene')

    # configure GPUs
    set_gpu(args.gpu)
    n_gpus = torch.cuda.device_count()

    # set up test folder
    test_path = os.path.join('ckpt', args.ckpt, 'test')
    ensure_path(test_path)

    rng = fix_random_seed(config.get('seed', 2022))

    ###########################################################################
    """ worker """

    ckpt_name = os.path.join(ckpt_path, 'last.pth')

    try:
        check_file(ckpt_name)
        ckpt = torch.load(ckpt_name)
        worker = RendererWorkerBase(
            cam_cfg=config.get('camera', ckpt['config']['camera']),
            model_cfg=ckpt['config']['model'],
        )
        worker.load(ckpt)
        worker.cuda(n_gpus)
    except:
        raise ValueError('checkpoint loading failed')

    yaml.dump(config, open(os.path.join(test_path, 'config.yaml'), 'w'))
    
    print('worker initialized')

    ############################################################################
    """ dataset """

    gt = make_images(config['target'])              # (v, 1/3, h, w)

    ############################################################################
    """ Validation """

    _, output_dict, metrics_dict = worker.eval(
        target=gt, 
        cfg=config['eval'],
    )

    pred = output_dict['pred'].numpy()
    target = output_dict['target'].numpy()
    pred = np.clip(pred, 0, 1)
    target = np.clip(target, 0, 1)
    pred = pred.transpose(0, 2, 3, 1)               # (v, h, w, 1/3)
    target = target.transpose(0, 2, 3, 1)           # (v, h, w, 1/3)

    imgs = np.concatenate([target, pred], axis=2)
    imgs = (imgs * 255).astype(np.uint8)
    if imgs.shape[-1] == 1:
        imgs = imgs[..., 0]

    for idx in range(len(imgs)):
        im = Image.fromarray(imgs[idx])
        im.save(os.path.join(test_path, '{:02d}.png'.format(idx)))

    print('RMSE: {:.3f}'.format(metrics_dict['rmse'].item()))
    print('PSNR: {:.3f}'.format(metrics_dict['psnr'].item()))
    print('SSIM: {:.3f}'.format(metrics_dict['ssim'].item()))

################################################################################
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-ckpt', '--ckpt', help='checkpoint folder')
    parser.add_argument('-c', '--config', help='config file')
    parser.add_argument('-g', '--gpu', type=str, default='0', 
                        help='GPU device IDs')
    args = parser.parse_args()

    main(args)