""" Per-scene RSD reconstruction """

import os
import argparse

import yaml
import numpy as np
from PIL import Image

import torch

from libs.config import load_config
from libs.data import make_measurement
from libs.encoder import make_rsd
from libs.utils import *


def main(args):
    # set up save folder
    save_path = os.path.join('./rsd', args.name)
    ensure_path(save_path)

    # load config file
    check_file(args.config)
    config = load_config(args.config, mode='rsd_per_scene')

    yaml.dump(config, open(os.path.join(save_path, 'config.yaml'), 'w'))

    # torch.backends.cudnn.enabled = True
    # torch.backends.cudnn.benchmark = True
    # torch.backends.cudnn.deterministic = True

    ############################################################################
    """ dataset"""

    x = make_measurement(config['dataset'])          # (1/3, t, h, w)
    
    ############################################################################
    """ Phasor field RSD reconstruction """
    
    rsd = make_rsd(config['rsd'], efficient=True)
    rsd.eval()
    with torch.no_grad():
        x = rsd(x.unsqueeze(0))
    x = x.squeeze(0).cpu().numpy()                   # (1/3, d, h, w)

    # maximum projection along depth axis
    x = x.max(1).transpose(1, 2, 0)                  # (h, w, 1/3)
    x = x / x.max()

    x = (x * 255).astype(np.uint8)
    if x.shape[-1] == 1:
        x = x[..., 0]
    im = Image.fromarray(x)
    im.save(os.path.join(save_path, 'recon.png'))

################################################################################
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', help='config file')
    parser.add_argument('-n', '--name', help='job name')
    args = parser.parse_args()

    main(args)