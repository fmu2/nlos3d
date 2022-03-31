import os
import random

import cv2
import json
import numpy as np
import scipy.io as sio
import torch
from torch.utils.data import Dataset

from .utils import check_file


class NLOSCompose:

    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, x):
        for t in self.transforms:
            x = t(x)
        return x

    def __repr__(self):
        repr_str = ''
        for t in self.transforms:
            repr_str += t.__repr__() + '\n'
        return repr_str


class NLOSPoissonNoise:

    def __init__(self, background=[0.05, 0.5]):
        self.rate = background

    def __call__(self, x):
        if isinstance(self.rate, (int, float)):
            rate = self.rate
        elif isinstance(self.rate, (list, tuple)):
            rate = random.random() * (self.rate[1] - self.rate[0]) 
            rate += self.rate[0]
        poisson = torch.distributions.Poisson(rate)
        # shot noise + background noise
        x = torch.poisson(x) + poisson.sample(x.shape)
        return x

    def __repr__(self):
        return "Introduce shot noise and background noise to raw histograms"


class NLOSRandomScale:

    def __init__(self, scale=1):
        self.scale = scale

    def __call__(self, x):
        if isinstance(self.scale, (int, float)):
            x *= self.scale
        elif isinstance(self.scale, (list, tuple)):
            scale = random.random() * (self.scale[1] - self.scale[0]) 
            scale += self.scale[0]
            x *= scale
        return x

    def __repr__(self):
        return "Randomly scale raw histograms"


def get_transform(scale=1, background=0):
    transform = [NLOSRandomScale(scale)]
    if background != 0:
        transform += [NLOSPoissonNoise(background)]
    transform = NLOSCompose(transform)
    return transform

def make_measurement(config):
    check_file(config["path"])

    try:
        x = sio.loadmat(
            config["path"], verify_compressed_data_integrity=False
        )["data"]
        if x.ndim == 3:
            x = x[None]
        x = x[:, :config["clip"]] * config["scale"]          # (1/3, t, h, w)
        
        if x.shape[0] == 3:
            assert config["color"] in ("rgb", "gray", "r", "g", "b"), \
                "invalid color: {:s}".format(config["color"])
            if config["color"] == "gray":
                x = 0.299 * x[0:1] + 0.587 * x[1:2] + 0.114 * x[2:3]
            elif config["color"] == "r":    x = x[0:1]
            elif config["color"] == "g":    x = x[1:2]
            elif config["color"] == "b":    x = x[2:3]
    except:
        raise ValueError("data loading failed: {:s}".format(config["path"]))
    
    x = torch.from_numpy(x.astype(np.float32))               # (1/3, t, h, w)
    x = get_transform(config["scale"], config["background"])(x)
    if torch.cuda.is_available():
        x = x.cuda(non_blocking=True)
    return x

def make_images(config):
    check_file(config["path"])
    
    try:
        x = sio.loadmat(
            config["path"], verify_compressed_data_integrity=False
        )["data"]
        x = cv2.resize(x, (config["target_size"],) * 2)      # (h, w, v*3)
        x = x.reshape(*x.shape[:2], x.shape[-1] // 3, 3)     # (h, w, v, 3)
        
        assert config["color"] in ("rgb", "gray", "r", "g", "b"), \
            "invalid color: {:s}".format(config["color"])
        if config["color"] == "gray":
            x = 0.299 * x[..., 0:1] + 0.587 * x[..., 1:2] + 0.114 * x[..., 2:3]
        elif config["color"] == "r":    x = x[..., 0:1]
        elif config["color"] == "g":    x = x[..., 1:2]
        elif config["color"] == "b":    x = x[..., 2:3]
    except:
        raise ValueError("data loading failed: {:s}".format(config["path"]))

    x = torch.from_numpy(x.astype(np.float32))               # (h, w, v, 1/3)
    if torch.cuda.is_available():
        x = x.cuda(non_blocking=True)
    return x

def make_depths(config):
    check_file(config["path"])

    try:
        x = sio.loadmat(
            config["path"], verify_compressed_data_integrity=False
        )["data"]
        x = cv2.resize(x, (config["target_size"],) * 2)      # (h, w, v)
    except:
        raise ValueError("data loading failed: {:s}".format(config["path"]))

    x = torch.from_numpy(x.astype(np.float32))               # (h, w, v)
    if torch.cuda.is_available():
        x = x.cuda(non_blocking=True)
    return x

def make_dataset(config, split=None):
    if split is None:
        return NLOSDataset(**config)
    else:
        return NLOSDataset(split=split, **config)

def collate_fn(data):
    meas, image, depth = zip(*data)
    meas = None if None in meas else torch.stack(meas)
    image = None if None in image else torch.stack(image)
    depth = None if None in depth else torch.stack(depth)
    return meas, image, depth

def cycle(iterable):
    while True:
        for x in iterable:
            yield x
            

class NLOSDataset(Dataset):

    def __init__(
        self, 
        root,               # dataset root directory
        split,              # data split ("train", "val")
        res="1cm",          # temporal resolution of histograms
        clip=512,           # time range of histograms
        scale=1,            # scaling factor (float or float tuple)
        background=0,       # background noise rate (float or float tuple)
        target_size=256,    # target image size (unit: px)
        target_noise=0,     # standard deviation of target image noise
        color="rgb",        # color channel(s) of target image
    ):
        super(NLOSDataset, self).__init__()

        self.root = root
        self.clip = clip
        self.transform = get_transform(scale, background)
        self.target_size = target_size
        self.target_noise = target_noise

        assert color in ("rgb", "gray", "r", "g", "b"), \
            "invalid color: {:s}".format(color)
        self.color = color

        split_file = os.path.join(
            root, "splits", "{:s}_{:s}.json".format(res, split)
        )
        check_file(split_file)

        with open(split_file, 'r') as f:
            data_list = json.load(f)
        self.data_list = data_list

    def _load_meas(self, idx):
        name = self.data_list[idx]["meas"]
        path = os.path.join(self.root, "data", name)
        check_file(path)
        
        try:
            x = sio.loadmat(
                path, verify_compressed_data_integrity=False
            )["data"]
            if x.ndim == 3:
                x = x[None]
            x = x[:, :self.clip]                                # (1/3, t, h, w)
            if x.shape[0] == 3:
                if self.color == "gray":
                    x = 0.299 * x[0:1] + 0.587 * x[1:2] + 0.114 * x[2:3]
                elif self.color == "r": x = x[0:1]
                elif self.color == "g": x = x[1:2]
                elif self.color == "b": x = x[2:3]
        except:
            raise ValueError("measurement loading failed: {:s}".format(path))
        
        x = torch.from_numpy(x.astype(np.float32))              # (1/3, t, h, w)
        return x

    def _load_image(self, idx):
        try:
            name = self.data_list[idx]["image"]
        except:
            return None
        
        path = os.path.join(self.root, "data", name)
        check_file(path)
        
        try:
            if name.split('.')[-1] == "mat":
                x = sio.loadmat(
                    path, verify_compressed_data_integrity=False
                )["data"]
            else:
                x = cv2.imread(path)[..., ::-1] / 255
            x = cv2.resize(x, (self.target_size,) * 2)          # (h, w, v*3)
            x = x.reshape(*x.shape[:2], x.shape[-1] // 3, 3)    # (h, w, v, 3)
            if self.color == "gray":
                x = 0.299 * x[..., 0:1] + 0.587 * x[..., 1:2] + 0.114 * x[..., 2:3]
            elif self.color == "r": x = x[..., 0:1]
            elif self.color == "g": x = x[..., 1:2]
            elif self.color == "b": x = x[..., 2:3]
        except:
            raise ValueError("image loading failed: {:s}".format(path))

        x = torch.from_numpy(x.astype(np.float32))              # (h, w, v, 1/3)
        x = x.permute(2, 3, 0, 1)                               # (v, 1/3, h, w)
        if self.target_noise > 0:
            x += torch.randn_like(x) * self.target_noise
            x = torch.clamp(x, min=0)
        return x

    def _load_depth(self, idx):
        try:
            name = self.data_list[idx]["depth"]
        except:
            return None

        path = os.path.join(self.root, "data", name)
        check_file(path)
        
        try:
            x = sio.loadmat(
                path, verify_compressed_data_integrity=False
            )["data"]
            x = cv2.resize(x, (self.target_size,) * 2)          # (h, w, v)
        except:
            raise ValueError("depth loading failed: {:s}".format(path))

        x = torch.from_numpy(x.astype(np.float32))              # (h, w, v)
        x = x.permute(2, 0, 1)                                  # (v, h, w)
        return x

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        meas = self.transform(self._load_meas(idx))
        images = self._load_image(idx)
        depths = self._load_depth(idx)
        return meas, images, depths