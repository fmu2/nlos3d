import os
import argparse

import yaml
import numpy as np
from PIL import Image

import torch
from torch.utils.data import DataLoader

from libs.config import load_config
from libs.data import collate_fn, make_dataset
from libs.worker import EncoderDecoderWorker
from libs.utils import *


def main(args):
    # fetch checkpoint folder
    ckpt_path = os.path.join('ckpt', args.ckpt)
    check_path(ckpt_path)

    # load config
    check_file(args.config)
    config = load_config(args.config, mode='test_nr')

    # configure GPUs
    set_gpu(args.gpu)
    n_gpus = torch.cuda.device_count()

    # set up test folder
    test_dir = os.path.join('ckpt', args.ckpt, 'test')
    os.makedirs(test_dir, exist_ok=True)
    test_path = os.path.join(test_dir, config['split'])
    ensure_path(test_path)

    rng = fix_random_seed(config.get('seed', 2022))

    ###########################################################################
    """ worker """

    ckpt_name = os.path.join(ckpt_path, 'last.pth')

    try:
        check_file(ckpt_name)
        ckpt = torch.load(ckpt_name)
        worker = EncoderDecoderWorker(ckpt['config']['model'])
        worker.load(ckpt)
        worker.cuda(n_gpus)
    except:
        raise ValueError('checkpoint loading failed')

    yaml.dump(config, open(os.path.join(test_path, 'config.yaml'), 'w'))
    
    print('worker initialized')

    ############################################################################
    """ dataset """

    dataset = make_dataset(config['dataset'], split=config['split'])
    loader = DataLoader(
        dataset, 
        batch_size=config['batch_size'],
        num_workers=config['n_workers'],
        collate_fn=collate_fn,
        shuffle=False, 
        pin_memory=True, 
        drop_last=False,
    )
    print('dataset size: {:d}'.format(len(dataset)))

    ############################################################################
    """ Training / Validation """

    metric_list = ['rmse', 'psnr', 'ssim']
    metrics = {k: AverageMeter() for k in metric_list}
    
    idx = 1
    for (meas, target, _) in loader:
        _, output_dict, metric_dict = worker.eval(
            meas=meas,
            target=target,
            cfg=config['eval'],
        )

        for k in metric_dict.keys():
            metrics[k].update(metric_dict[k].item())

        pred = output_dict['pred'].numpy()          # (bs, 1/3, h, w)
        target = output_dict['target'].numpy()      # (bs, 1/3, h, w)
        pred = pred.transpose(0, 2, 3, 1)           # (bs, h, w, 1/3)
        target = target.transpose(0, 2, 3, 1)       # (bs, h, w, 1/3)
        
        imgs = np.concatenate([target, pred], axis=2)
        imgs = (imgs * 255).astype(np.uint8)
        if imgs.shape[-1] == 1:
            imgs = imgs[..., 0]

        for i in range(len(imgs)):
            im = Image.fromarray(imgs[i])
            im.save(os.path.join(test_path, '{:04d}.png'.format(idx)))
            idx += 1

    print('RMSE: {:.3f}'.format(metrics['rmse'].item()))
    print('PSNR: {:.3f}'.format(metrics['psnr'].item()))
    print('SSIM: {:.3f}'.format(metrics['ssim'].item()))

################################################################################
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-ckpt', '--ckpt', help='checkpoint folder')
    parser.add_argument('-c', '--config', help='config file')
    parser.add_argument('-g', '--gpu', type=str, default='0', 
                        help='GPU device IDs')
    args = parser.parse_args()

    main(args)