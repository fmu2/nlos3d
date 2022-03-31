import os
import time
import shutil
import random

import torch
import torch.backends.cudnn as cudnn
import numpy as np

_log_path = None

def set_gpu(gpu):
    print('set gpu:', gpu)
    os.environ['CUDA_VISIBLE_DEVICES'] = gpu

def check_file(path):
    if not os.path.isfile(path):
        raise ValueError('file does not exist: %s' % path)

def check_path(path):
    if not os.path.exists(path):
        raise ValueError('path does not exist: %s' % path)

def ensure_path(path, remove=True):
    basename = os.path.basename(path.rstrip('/'))
    if os.path.exists(path):
        return True
        # if remove and (basename.startswith('_')
        #   or input('{} exists, remove? ([y]/n): '.format(path)) != 'n'):
        #   shutil.rmtree(path)
        #   os.makedirs(path)
    else:
        os.makedirs(path)
        return False

def set_log_path(path):
    global _log_path
    _log_path = path

def log(obj, filename='log.txt'):
    print(obj)
    if _log_path is not None:
        with open(os.path.join(_log_path, filename), 'a') as f:
            print(obj, file=f)

def count_params(model, return_str=True):
    n_params = 0
    for p in model.parameters():
        n_params += p.numel()
    if return_str:
        if n_params >= 1e6:
            return '{:.1f}M'.format(n_params / 1e6)
        else:
            return '{:.1f}K'.format(n_params / 1e3)
    else:
        return n_params

def time_str(t):
    if t >= 3600:
        return '{:.1f}h'.format(t / 3600)
    if t >= 60:
        return '{:.1f}m'.format(t / 60)
    return '{:.1f}s'.format(t)
    

class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0.
        self.avg = 0.
        self.sum = 0.
        self.count = 0.

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def item(self):
        return self.avg


class Timer(object):
    def __init__(self):
        self.start()

    def start(self):
        self.v = time.time()

    def end(self):
        return time.time() - self.v


def fix_random_seed(seed, reproduce=False):
    cudnn.enabled = True
    cudnn.benchmark = True
    
    if reproduce:
        cudnn.benchmark = False
        cudnn.deterministic = True
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        ## NOTE: uncomment for CUDA >= 10.2
        # os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
        ## NOTE: uncomment for pytorch >= 1.8
        # torch.use_deterministic_algorithms(True)

    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    rng = torch.manual_seed(seed)

    return rng