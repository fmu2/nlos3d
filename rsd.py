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

    dataset = make_dataset(config['dataset'])
    print('dataset size: {:d}'.format(len(dataset)))

    loader = DataLoader(
        dataset, 
        batch_size=config['batch_size'],
        num_workers=12,
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

    print(save_path)
    print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
    idx = 1
    for (meas, target, _) in loader:
        meas = meas.cuda(non_blocking=True)
        if target is not None:
            target = target[:, -1]

        with torch.no_grad():
            x = rsd(meas)                       # (b, 1/3, d, h, w)
        x = x.cpu().numpy()

        # maximum projection along depth axis
        x = x.max(2)                            # (b, 1/3, h, w)
        x = x / x.max((1, 2, 3), keepdims=True)
        x = x.transpose(0, 2, 3, 1)             # (b, h, w, 1/3)

        if target is not None:
            target = target.numpy().transpose(0, 2, 3, 1)
            x = np.concatenate([target, x], axis=2)
        x = (x * 255).astype(np.uint8)
        if x.shape[-1] == 1:
            x = x[..., 0]

        for i in range(len(x)):
            im = Image.fromarray(x[i])
            im.save(os.path.join(save_path, ('{:04d}.png').format(idx)))
            idx += 1

################################################################################
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', help='config file')
    parser.add_argument('-n', '--name', help='job name')
    parser.add_argument('-g', '--gpu', help='GPU device IDs', 
                        type=str, default='0')
    args = parser.parse_args()

    main(args)

    print("done!")