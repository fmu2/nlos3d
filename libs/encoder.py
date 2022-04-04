import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.fft as fft

import numpy as np
import scipy.signal as ssi

from .modules import *


class RSDBase(nn.Module):
    """ Rayleigh-Sommerfield diffraction kernel """
    def __init__(
        self, 
        t=256,              # time dimension of input volume
        d=32,               # depth dimension of output volume
        h=64,               # height dimension of input/output volume
        w=64,               # width dimension of input/output volume
        in_plane=6,         # number of input planes
        wall_size=2,        # wall size (unit: m)
        bin_len=0.02,       # distance covered by a bin (unit: m)
        zmin=0,             # min reconstruction depth w.r.t. the wall (unit: m)
        zmax=2,             # max reconstruction depth w.r.t. the wall (unit: m)
        scale_coef=1,       # scale coefficient for virtual wavelength
        n_cycles=4,         # number of cycles for virtual wavelet
        ratio=0.1,          # relative magnitude under which a frequency is discarded
        actv="linear",      # activation function
        norm="max",         # normalization function
        affine=False,       # if True, apply a learnable affine transform in norm
        efficient=False,    # if True, use memory-efficient implementation
        **kwargs,
    ):
        super(RSDBase, self).__init__()
        assert t % 2 == 0, "time dimension must be even"

        self.t = t
        self.d = d
        self.h = h
        self.w = w
        self.in_plane = in_plane
        self.out_plane = in_plane

        self.wall_size = wall_size
        self.bin_len = bin_len
        self.zmin = zmin
        self.zmax = zmax

        self.scale_coef = scale_coef
        self.n_cycles = n_cycles
        self.ratio = ratio

        bin_resolution = bin_len / 3e8       # temporal bin resolution (unit: sec)
        sampling_freq = 1 / bin_resolution   # temporal sampling frequency

        # define virtual wave
        wall_spacing = wall_size / h         # sample spacing on the wall (unit: m)
        lambda_limit = 2 * wall_spacing      # smallest achievable wavelength
        wavelength = scale_coef * lambda_limit

        wave = self._define_wave(wavelength)
        fwave = np.abs(np.fft.fft(wave) / t)[:len(wave) // 2 + 1]
        coef_ratio = fwave / np.max(fwave)

        # retain spectrum [lambda - delta, lambda + delta]
        freq_idx = np.where(coef_ratio > ratio)[0]
        print(
            "{:d}/{:d} frequencies kept in RSD".format(
                len(freq_idx), len(fwave)
            )
        )
        freqs = sampling_freq * freq_idx / t
        omegas = 2 * np.pi * freqs           # angular frequencies
        coefs = fwave[freq_idx]              # weight cofficients

        # define RSD kernel
        zdim = np.linspace(zmin, zmax, d + 1)
        zdim = (zdim[:-1] + zdim[1:]) / 2    # mid-point rule

        rsd, tgrid = self._define_rsd(zdim, omegas)
        if not efficient:
            rsd = np.pad(rsd, ((0, 0), (0, 0), (0, h), (0, w))) # (o, d, h*2, w*2)
        frsd = np.fft.fft2(rsd)                                 # (o, d, h(*2), w(*2))
        
        # define phase term in IFFT
        omegas = omegas.reshape(-1, 1, 1, 1)                    # (o, 1, 1, 1)
        tgrid = (zdim / 3e8).reshape(1, -1, 1, 1)               # (1, d, 1, 1)
        phase = np.exp(1j * omegas * tgrid)                     # (o, d, h/1, w/1)

        # parameters associated with virtual wave
        freq_idx = torch.from_numpy(freq_idx)                   # (o,)
        self.register_buffer("freq_idx", freq_idx, persistent=False)

        coefs = torch.from_numpy(coefs.astype(np.float32))
        coefs = coefs.reshape(-1, 1, 1)                         # (o, 1, 1)
        self.register_buffer("coefs", coefs, persistent=False)

        # parameters associated with RSD propagation
        frsd = torch.from_numpy(frsd.astype(np.complex64))      # (o, d, h(*2), w(*2))
        self.register_buffer("frsd", frsd, persistent=False)
        
        phase = torch.from_numpy(phase.astype(np.complex64))    # (o, d, h, w)
        self.register_buffer("phase", phase, persistent=False)

        self.actv = make_actv(actv)
        self.norm = make_norm3d(norm, in_plane, affine)

    def _define_wave(self, wavelength):
        # discrete samples of the virtual wavelet
        samples = round((self.n_cycles * wavelength) / self.bin_len)
        n_cycles = samples * self.bin_len / wavelength
        idx = np.arange(samples) + 1

        # complex-valued sinusoidal wave modulated by gaussian envelope
        sinusoid = np.exp(1j * 2 * np.pi * n_cycles * idx / samples)
        win = ssi.gaussian(samples, (samples - 1) / 2 * 0.3)
        wave = sinusoid * win

        # pad wave to the same length as time-domain histograms
        if len(wave) < self.t:
            wave = np.pad(wave, (0, self.t - len(wave)))
        return wave

    def _define_rsd(self, zdim, omegas):
        width = self.wall_size / 2
        ydim = np.linspace(width, -width, self.h + 1)
        xdim = np.linspace(-width, width, self.w + 1)
        ydim = (ydim[:-1] + ydim[1:]) / 2   # mid-point rule
        xdim = (xdim[:-1] + xdim[1:]) / 2
        [zgrid, ygrid, xgrid] = np.meshgrid(zdim, ydim, xdim, indexing="ij")

        # a grid of distance between wall center and scene points
        # (assume light source lies at wall center)
        dgrid = np.sqrt((xgrid ** 2 + ygrid ** 2) + zgrid ** 2) # (d, h, w)
        tgrid = zgrid / 3e8                                     # (d, h, w)

        # RSD kernel (falloff term is ignored)
        dgrid = dgrid.reshape(1, len(zdim), self.h, self.w)     # (1, d, h, w)
        omegas = omegas.reshape(-1, 1, 1, 1)                    # (o, 1, 1, 1)
        rsd = np.exp(1j * omegas / 3e8 * dgrid) / dgrid         # (o, d, h, w)
        return rsd, tgrid

    def forward(self, x, sqrt=True):
        raise NotImplementedError("RSD forward pass not implemented")


class RSD(RSDBase):

    def __init__(self, **kwargs):
        super(RSD, self).__init__(**kwargs)

    def forward(self, x, sqrt=True):
        """
        Args:
            x (float tensor, (bs, c, t, h, w)): input time-domain features.
            sqrt (bool): if True, take the square root before normalization.

        Returns:
            x (float tensor, (bs, c, d, h, w)): output space-domain features.
        """
        bs, c, t, h, w = x.shape
        assert t == self.t, \
            "time dimension should be {:d}, got {:d}".format(self.t, t)
        assert h == self.h, \
            "height dimension should be {:d}, got {:d}".format(self.h, h)
        assert w == self.w, \
            "width dimension should be {:d}, got {:d}".format(self.w, w)
        assert c == self.in_plane, \
            "feature dimension should be {:d}, got {:d}".format(self.in_plane, c)

        # propagate each feature dimension independently
        tdata = x.flatten(0, 1)                         # (bs*c, t, h, w)

        ## Step 1: convert measurement into FDH
        fdata = fft.rfft(tdata, dim=1)                  # (bs*c, t//2+1, h, w)
        fdata = fdata[:, self.freq_idx]                 # (bs*c, o, h, w)

        ## Step 2: define source phasor field
        phasor = self.coefs * fdata                     # (bs*c, o, h, w)
        phasor = F.pad(phasor, (0, w, 0, h))            # (bs*c, o, h*2, w*2)
        fsrc = fft.fftn(phasor, s=[-1, -1])             # (bs*c, o, h*2, w*2)

        ## Step 3: RSD propagation
        # WARNING: PyTorch is buggy when distributing complex tensors
        # here is a temporary workaround
        frsd, phase = self.frsd, self.phase
        if frsd.dim() == 5:
            frsd = torch.complex(frsd[..., 0], frsd[..., 1])
        if phase.dim() == 5:
            phase = torch.complex(phase[..., 0], phase[..., 1])
        fdst = fsrc.unsqueeze(2) * frsd
        # fdst = fsrc.unsqueeze(2) * self.frsd
        fdst = phase * fdst
        # fdst = self.phase * fdst                        # (bs*c, o, d, h*2, w*2)
        fvol = torch.sum(fdst, 1)                       # (bs*c, d, h*2, w*2)
        tvol = fft.ifftn(fvol, s=[-1, -1])
        tvol = tvol[:, :, h//2:h + h//2, w//2:w + w//2] # (bs*c, d, h, w)
        
        ## Step 4: post-process data
        tvol = torch.abs(tvol)                          # (bs*c, d, h, w)
        if not sqrt:
            tvol = tvol ** 2

        x = tvol.reshape(bs, c, self.d, h, w)
        x = self.actv(self.norm(x))
        return x


class RSDEfficient(RSDBase):
    """
    NOTE: this implementation does not zero-pad RSD kernel for efficiency.
    This results in sparser frequency sampling (4x memory saving) and 
    slightly noiser reconstruction results (with aliasing).
    """
    def __init__(self, **kwargs):
        super(RSDEfficient, self).__init__(efficient=True, **kwargs)

    def forward(self, x, sqrt=True):
        """
        Args:
            x (float tensor, (bs, c, t, h, w)): input time-domain features.
            sqrt (bool): if True, take the square root before normalization.

        Returns:
            x (float tensor, (bs, c, d, h, w)): output space-domain features.
        """
        bs, c, t, h, w = x.shape
        assert t == self.t, \
            "time dimension should be {:d}, got {:d}".format(self.t, t)
        assert h == self.h, \
            "height dimension should be {:d}, got {:d}".format(self.h, h)
        assert w == self.w, \
            "width dimension should be {:d}, got {:d}".format(self.w, w)
        assert c == self.in_plane, \
            "feature dimension should be {:d}, got {:d}".format(self.in_plane, c)

        # propagate each feature dimension independently
        tdata = x.flatten(0, 1)                         # (bs*c, t, h, w)

        ## Step 1: convert measurement into FDH
        fdata = fft.rfft(tdata, dim=1)                  # (bs*c, t//2+1, h, w)
        fdata = fdata[:, self.freq_idx]                 # (bs*c, o, h, w)

        ## Step 2: define source phasor field
        phasor = self.coefs * fdata                     # (bs*c, o, h, w)
        fsrc = fft.fftn(phasor, s=[-1, -1])             # (bs*c, o, h, w)

        ## Step 3: RSD propagation
        # WARNING: PyTorch is buggy when distributing complex tensors
        # here is a temporary workaround
        frsd, phase = self.frsd, self.phase
        if frsd.dim() == 5:
            frsd = torch.complex(frsd[..., 0], frsd[..., 1])
        if phase.dim() == 5:
            phase = torch.complex(phase[..., 0], phase[..., 1])
        fdst = fsrc.unsqueeze(2) * frsd
        # fdst = fsrc.unsqueeze(2) * self.frsd
        fdst = phase * fdst                             # (bs*c, o, d, h, w)
        # fdst = self.phase * fdst                        # (bs*c, o, d, h, w)
        fvol = torch.sum(fdst, 1)                       # (bs*c, d, h, w)
        tvol = fft.ifftn(fvol, s=[-1, -1])
        tvol = fft.ifftshift(tvol, dim=(-2, -1))
        
        ## Step 4: post-process data
        tvol = torch.abs(tvol)                          # (bs*c, d, h, w)
        if not sqrt:
            tvol = tvol ** 2

        x = tvol.reshape(bs, c, self.d, h, w)
        x = self.actv(self.norm(x))
        return x


class FFCNet(nn.Module):

    def __init__(
        self, 
        t=512,
        d=64,               # depth dimension of output volume
        h=128,              # height dimension of input volume
        w=128,              # width dimension of input volume
        in_plane=1,         # number of input planes
        plane=256,          # number of planes prior to propagation
        out_plane=3,        # number of output planes
        n_layers=5,         # number of FFC blocks
        bottleneck=True,    # if True, use bottleneck blocks
        expansion=4,        # expansion factor for bottleneck
        actv="relu",        # activation function
        norm="batch",       # normalization function
        affine=True,        # if True, apply learnable affine transform in norm
        pe=False,           # if True, apply position encoding
        **kwargs,
    ):
        super(FFCNet, self).__init__()
        assert t % 2 == 0, "time dimension must be even"
        assert h % 2 == 0, "height dimension must be even"
        assert w % 2 == 0, "weight dimension must be even"

        self.t = t
        self.d = d
        self.h = h
        self.w = w

        self.in_plane = in_plane
        self.plane = plane
        self.out_plane = out_plane

        self.in_ffc = FFC2d(
            in_plane=in_plane * (t // 2 + 1), 
            plane=plane, 
            actv=actv, 
            norm=norm, 
            affine=affine,
            pe=pe,
        )

        self.blocks = ResBlockFFC2d(
            plane=plane, 
            n_layers=n_layers, 
            bottleneck=bottleneck,
            expansion=expansion, 
            actv=actv, 
            norm=norm, 
            affine=affine,
            pe=pe,
        )
        self.out_conv = nn.Conv2d(plane, out_plane * d, 3, 1, 1)

    def forward(self, x):
        bs, c, t, h, w = x.shape
        assert t == self.t, \
            "time dimension should be {:d}, got {:d}".format(self.t, t)
        assert h == self.h, \
            "height dimension should be {:d}, got {:d}".format(self.h, h)
        assert w == self.w, \
            "width dimension should be {:d}, got {:d}".format(self.w, w)
        assert c == self.in_plane, \
            "feature dimension should be {:d}, got {:d}".format(self.in_plane, c)

        x = fft.rfft(x, dim=2)          # (bs, ci, o, h, w)
        x = x.flatten(1, 2)             # (bs, ci * o, h, w)

        x = self.in_ffc(x)              # (bs, c, h, w)
        x = x.reshape(bs, self.plane, h // 2, 2, w // 2, 2)
        x = x.mean(dim=(-3, -1))        # (bs, c, h/2, w/2)

        x = self.blocks(x)              # (bs, c, h/2, w/2)
        x = torch.abs(x)
        x = self.out_conv(x)            # (bs, co * d, h/2, w/2)

        x = x.reshape(bs, self.out_plane, self.d, *x.shape[-2:])
        return x


class FRN(nn.Module):
    """
    Feature extraction and propagation network
    (NOTE: this implementation strictly follows Chen et al., SIGGRAPH Asia 2020
     for reproducibility)
    """
    def __init__(
        self,
        in_plane=1,         # number of input planes
        plane=6,            # number of planes prior to propagation
        rsd_layer=None,     # RSD kernel
        actv="leaky_relu",  # activation function
        norm="none",        # normalization function
        affine=False,       # if True, apply learnable affine transform in norm
        **kwargs,
    ):
        super(FRN, self).__init__()

        bias = True if norm == "none" or not affine else False

        # conv1 is initialized as a mean filter
        conv1 = torch.zeros(1, in_plane, 3, 3, 3)
        conv1[..., 1:, 1:, 1:] = 0.125
        self.conv1 = nn.Parameter(conv1)

        self.conv2 = nn.Sequential(
            nn.Conv3d(in_plane, plane - 1, 3, 2, 1),
            ResBlock3d(plane - 1, plane - 1, 1, 2, actv, norm, affine),
        )

        self.rsd_layer = rsd_layer

    def forward(self, x):
        x1 = F.conv3d(x, self.conv1, bias=None, stride=2, padding=1)
        x2 = self.conv2(x)
        x = torch.cat([x1, x2], dim=1)
        x = self.rsd_layer(x)
        return x


class RSDNet(nn.Module):
    """ RSD encoder """
    def __init__(
        self, 
        in_plane=1,         # number of input planes 
        plane=6,            # number of planes prior to propagation
        in_block=True,      # if True, learn conv block before RSD
        ds=False,           # if True, down-sample the input
        rsd_layer=None,     # RSD kernel
        actv="leaky_relu",  # activation function
        norm="none",        # normalization function
        affine=False,       # if True, apply learnable affine transform in norm
        **kwargs,
    ):
        super(RSDNet, self).__init__()

        bias = True if norm == "none" or not affine else False
        stride = 2 if ds else 1

        if in_block:
            self.in_block = nn.Sequential(
                nn.Conv3d(in_plane, plane, 3, stride, 1, bias=bias),
                make_norm3d(norm, plane, affine),
                make_actv(actv),
                nn.Conv3d(plane, plane, 3, 1, 1, bias=bias),
                make_norm3d(norm, plane, affine),
                make_actv(actv),
            )
            in_plane = plane
        else:
            self.in_block = nn.Identity()

        self.rsd_layer = rsd_layer
        self.out_block = nn.Sequential(
            nn.Conv3d(in_plane, plane, 3, 1, 1, bias=bias),
            make_norm3d(norm, plane, affine),
            make_actv(actv),
            nn.Conv3d(plane, plane, 3, 1, 1),
        )

    def forward(self, x):
        x = self.in_block(x)
        x = self.rsd_layer(x)
        x = self.out_block(x)
        return x


class UNet(nn.Module):
    """ UNet encoder """
    def __init__(
        self, 
        in_plane=1,             # number of input planes
        plane=6,                # number of planes at top level
        max_plane=48,           # max number of planes
        n_levels=4,             # number of levels
        skip_layers=None,       # skip layers
        down_actv="leaky_relu", # activation function in top-down path
        up_actv="relu",         # activation function in bottom-up path
        norm="none",            # normalization function
        affine=False,           # if True, apply learnable affine transform in norm
        **kwargs,
    ):
        super(UNet, self).__init__()
        assert n_levels > 0, "need at least one pyramid level"
        if skip_layers is not None:
            assert len(skip_layers) == n_levels, \
                "number of skip layers mismatch"
        else:
            skip_layers = [None] * n_levels
        bias = True if norm == "none" or not affine else False

        self.in_block = nn.Sequential(
            nn.Conv3d(in_plane, plane, 3, 1, 1, bias=bias),
            make_norm3d(norm, plane, affine),
            make_actv(down_actv),
            nn.Conv3d(plane, plane, 3, 1, 1, bias=bias),
            make_norm3d(norm, plane, affine),
            make_actv(down_actv),
        )

        unet = UNetBlock(
            min(2 ** (n_levels - 1) * plane, max_plane),
            min(2 ** (n_levels - 1) * plane, max_plane),
            None, skip_layers[-1], down_actv, up_actv, norm="none"
        )
        for i in range(n_levels - 2, -1, -1):
            unet = UNetBlock(
                min(2 ** i * plane, max_plane),
                min(2 ** (i + 1) * plane, max_plane),
                unet, skip_layers[i], down_actv, up_actv, norm, affine
            )
        self.unet = nn.Sequential(
            unet,
            nn.Conv3d(2 * plane, plane, 3, 1, 1, bias=bias),
            make_norm3d(norm, plane, affine),
            make_actv(up_actv),
        )
        
        self.out_block = nn.Sequential(
            nn.Conv3d(plane, plane, 3, 1, 1, bias=bias),
            make_norm3d(norm, plane, affine),
            make_actv(up_actv),
            nn.Conv3d(plane, plane, 3, 1, 1),
        )

    def forward(self, x): 
        x = self.in_block(x)
        x = self.unet(x)
        x = self.out_block(x)
        return x


def make_rsd(config, efficient=True):
    if efficient:
        rsd = RSDEfficient(**config)
    else:
        rsd = RSD(**config)
    if torch.cuda.is_available():
        rsd.cuda()
    return rsd

def make_ffc(config):
    ffc = FFCNet(**config)
    return ffc

def make_frn(config, efficient=True):
    rsd = make_rsd(config["rsd"], efficient)
    rsd_layer = rsd
    frn = FRN(rsd_layer=rsd_layer, **config)
    return frn

def make_rsdnet(config, efficient=True):
    rsd = make_rsd(config["rsd"], efficient)
    rsd_layer = rsd
    rsdnet = RSDNet(rsd_layer=rsd_layer, **config)
    return rsdnet

def make_unet(config, efficient=True):
    if config["skip"] is not None:
        assert isinstance(config["skip"], (list, tuple)), \
            "skip layers must be specified as a list or tuple"
        skip_layers = []
        for s in config["skip"]:
            skip_layers.append(make_rsd(config[s], efficient))

    unet = UNet(skip_layers=skip_layers, **config)
    return unet

def make_encoder(config, efficient=True):
    if config is None:
        return None

    if config["type"] == "rsd":
        encoder = make_rsd(config["rsd"], efficient)
    elif config["type"] == "ffc":
        encoder = make_ffc(config["ffc"])
    elif config["type"] == "frn":
        encoder = make_frn(config["frn"], efficient)
    elif config["type"] == "rsdnet":
        encoder = make_rsdnet(config["rsdnet"], efficient)
    elif config["type"] == "unet":
        encoder = make_unet(config["unet"], efficient)
    else:
        raise ValueError("invalid encoder: {:s}".format(config["type"]))
    
    if torch.cuda.is_available():
        encoder.cuda()
    return encoder