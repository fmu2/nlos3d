import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

from .modules import *


class Transformer(nn.Module):
    """
    A transformation module that re-samples 3D feature volume given 
    arbitrary camera pose.
    """
    def __init__(
        self, 
        d,              # depth dimension of feature volume
        h,              # height dimension of feature volume
        w,              # width dimension of feature volume
        wall_size=2,    # wall size (unit: m)
        zmin=-1,        # near plane w.r.t. world frame (unit: m)
        zmax=1,         # far plane w.r.t. world frame (unit: m)
    ):
        super(Transformer, self).__init__()
        assert zmax > zmin, \
            "far plane depth must be larger than near plane depth"

        self.d = d
        self.h = h
        self.w = w

        # sampling grid
        width = height = wall_size / 2
        zdim = np.linspace(zmin, zmax, d + 1)
        ydim = np.linspace(height, -height, h + 1)
        xdim = np.linspace(-width, width, w + 1)
        zdim = (zdim[:-1] + zdim[1:]) / 2
        ydim = (ydim[:-1] + ydim[1:]) / 2
        xdim = (xdim[:-1] + xdim[1:]) / 2
        [zgrid, ygrid, xgrid] = np.meshgrid(zdim, ydim, xdim, indexing="ij")
        grid = np.stack([xgrid, ygrid, zgrid], -1)         # (d, h, w, 3)
        grid = grid.reshape(-1, 3).T                       # (3, d*h*w)
        self.register_buffer("grid", torch.from_numpy(grid.astype(np.float32)))
        
        bb_ctr = [0, 0, (zmax + zmin) / 2]
        bb_radius = [width, height, (zmax - zmin) / 2]
        self.register_buffer("bb_ctr", torch.Tensor(bb_ctr))
        self.register_buffer("bb_radius", torch.Tensor(bb_radius))

    def forward(self, x, rot):
        """
        Args:
            x (float tensor, (bs, c, d, h, w)): feature volume aligned with the 
                orthogonal view.
            rot (float tensor, (3, 3)): rotation matrix.

        Returns:
            x (float tensor, (bs, c, d, h, w)): re-sampled feature volume.
        """
        bs, _, d, h, w = x.shape
        assert [d, h, w] == [self.d, self.h, self.w]

        # rotate grid to align with the camera view
        p = torch.matmul(rot, self.grid).T                 # (d*h*w, 3)

        # normalize grid points to [-1, 1]
        ## NOTE: F.grid_sample assumes that y-axis points DOWNWARD
        p = p.sub_(self.bb_ctr).div_(self.bb_radius)
        p[:, 1].mul_(-1)     # flip y-axis
        p = p.reshape(d, h, w, 3).repeat(bs, 1, 1, 1, 1)   # (bs, d, h, w, 3)

        # trilinear interpolation
        x = F.grid_sample(x, p, align_corners=False)       # (bs, c, d, h, w)
        return x


class Projector(nn.Module):
    """
    A projection module that projects 3D feature volume into 2D feature maps.
    """
    def __init__(self):
        super(Projector, self).__init__()

    def forward(self, x):
        x, idx = x.max(2)
        d = x.size(2)
        # larger value for closer planes
        depth = (d - 1 - idx.float()) / (d - 1)
        return x, depth


class RendererV0(nn.Module):
    """
    A rendering module that maps 2D spatial domain features to an image.
    (NOTE: this implementation strictly follows Chen et al., SIGGRAPH Asia 2020 
     for reproducibility)
    """
    def __init__(
        self, 
        in_plane,
        out_plane=1,
        actv="leaky_relu",
        norm="none",
        affine=False,
    ):
        super(Renderer, self).__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_plane, in_plane, 3, 1, 1),
            ResBlock2d(in_plane, in_plane, 1, 2, actv, norm, affine),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_plane + 1, in_plane * 2, 3, 1, 1),
            ResBlock2d(in_plane * 2, in_plane * 2, 1, 2, actv, norm, affine),
        )
        self.out_conv = nn.Conv2d(in_plane * 2, out_plane, 3, 1, 1)

    def forward(self, x):
        x0 = F.interpolate(
            x[:, :1], scale_factor=2, mode="bilinear", align_corners=False
        )
        x = F.interpolate(
            self.conv1(x), scale_factor=2, mode="bilinear", align_corners=False
        )
        x = self.conv2(torch.cat([x0, x], dim=1))
        x = x0 + self.out_conv(x)
        return x


class RendererV1(nn.Module):
    """
    A rendering module that maps 2D spatial domain features to an image.
    """
    def __init__(
        self, 
        in_plane,           # number of input planes
        out_plane=1,        # number of output planes
        n_levels=1,         # number of up-sampling levels
        n_layers=2,         # number of layers in residual blocks
        min_plane=8,        # minimum number of planes
        actv="relu",        # activation function
        out_actv="linear",  # output transform
        norm="none",        # normalization function
        affine=False,       # if True, apply learnable affine transform in norm
    ):
        super(RendererV1, self).__init__()
        assert n_levels > 0, "need at least one upsampling level"

        self.n_levels = n_levels
        bias = True if norm == "none" or not affine else False

        self.in_conv = nn.Conv2d(in_plane, out_plane, 3, 1, 1)

        base_convs = []
        for _ in range(n_levels):
            base_convs.append(
                UpSample2d(out_plane, out_plane, actv="linear", norm="none")
            )
        self.base_convs = nn.ModuleList(base_convs)

        up_convs, res_convs = [], []
        plane = in_plane
        for _ in range(n_levels):
            up_convs.append(
                nn.Sequential(
                    ResBlock2d(in_plane, plane, 1, n_layers, actv, norm, affine),
                    UpSample2d(plane, plane, actv="linear", norm="none"),
                )
            )
            res_convs.append(nn.Conv2d(plane, out_plane, 3, 1, 1))
            in_plane = plane
            plane = max(in_plane // 2, min_plane)
        
        self.up_convs = nn.ModuleList(up_convs)
        self.res_convs = nn.ModuleList(res_convs)
        self.out_actv = make_actv(out_actv)

    def forward(self, x):
        bx = self.in_conv(x)
        for i in range(self.n_levels):
            x = self.up_convs[i](x)
            bx = self.base_convs[i](bx) + self.res_convs[i](x)
        x = self.out_actv(bx)
        return x


class RendererV2(nn.Module):
    """
    A rendering module that maps 2D spatial domain features to an image.
    """
    def __init__(
        self, 
        in_plane,           # number of input planes
        out_plane=1,        # number of output planes
        n_levels=1,         # number of up-sampling levels
        min_plane=8,        # minimum number of planes
        actv="relu",        # activation function
        out_actv="linear",  # output transform
        norm="none",        # normalization function
        affine=False,       # if True, apply learnable affine transform in norm
    ):
        super(RendererV2, self).__init__()
        assert n_levels > 0, "need at least one upsampling level"
        
        self.n_levels = n_levels
        self.in_conv = nn.Conv2d(in_plane, out_plane, 3, 1, 1)

        base_convs = []
        for _ in range(n_levels):
            base_convs.append(
                UpSample2d(out_plane, out_plane, actv="linear", norm="none")
            )
        self.base_convs = nn.ModuleList(base_convs)

        up_convs, res_convs = [], []
        for i in range(n_levels):
            plane = max(in_plane // 2, min_plane)
            up_convs.append(
                UpSample2d(in_plane, plane, actv=actv, norm=norm, affine=affine)
            )
            res_convs.append(nn.Conv2d(plane, out_plane, 3, 1, 1))
            in_plane = plane
        
        self.up_convs = nn.ModuleList(up_convs)
        self.res_convs = nn.ModuleList(res_convs)
        self.out_actv = make_actv(out_actv)

    def forward(self, x):
        bx = self.in_conv(x)
        for i in range(self.n_levels):
            x = self.up_convs[i](x)
            bx = self.base_convs[i](bx) + self.res_convs[i](x)
        x = self.out_actv(bx)
        return x


class Decoder(nn.Module):

    def __init__(self, transformer, projector, renderer):
        super(Decoder, self).__init__()

        self.transformer = transformer
        self.projector = projector
        self.renderer = renderer

    def forward(self, x, rot=None):
        """
        Args:
            x (float tensor, (bs, c, d, hi, wi)): feature volume.
            rot (float tensor, (v, 3, 3)): rotation matrices.

        Returns:
            x (float tensor, (bs, v, 1/3, ho, wo)): rendered images.
        """
        bs = x.size(0)

        if rot is not None:
            if rot.dim() == 2:
                rot = rot.unsqueeze(0)
            views = []
            for r in rot:
                views.append(self.transformer(x, r)) # (bs, c, d, hi, wi)
            x = torch.stack(views, 1).flatten(0, 1)  # (bs*v, c, d, hi, wi)

        x, _ = self.projector(x)                     # (bs*v, c, hi, wi)
        x = self.renderer(x)                         # (bs*v, c, ho, wo)
        x = x.reshape(bs, -1, *x.shape[-3:])         # (bs, v, 1/3, ho, wo)
        return x


def make_decoder(config):
    if config is None:
        return None

    cf = config["renderer"]
    if cf["type"] == "v0":
        renderer = RendererV0(**cf["v0"])
    elif cf["type"] == "v1":
        renderer = RendererV1(**cf["v1"])
    elif cf["type"] == "v2":
        renderer = RendererV2(**cf["v2"])
    else:
        raise ValueError("invalid renderer: {:s}".format(cf["type"]))

    decoder = Decoder(
        transformer=Transformer(**config["transformer"]), 
        projector=Projector(), 
        renderer=renderer,
    )
    if torch.cuda.is_available():
        decoder.cuda()
    return decoder