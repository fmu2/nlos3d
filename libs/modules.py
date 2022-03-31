import torch
import torch.nn as nn
import torch.nn.functional as F


def make_actv(actv):
    if actv == 'relu':
        return nn.ReLU(inplace=True)
    elif actv == 'leaky_relu':
        return nn.LeakyReLU(0.2, inplace=True)
    elif actv == 'exp':
        return lambda x: torch.exp(x)
    elif actv == 'sigmoid':
        return lambda x: torch.sigmoid(x)
    elif actv == 'tanh':
        return lambda x: torch.tanh(x)
    elif actv == 'softplus':
        return lambda x: torch.log(1 + torch.exp(x - 1))
    elif actv == 'linear':
        return nn.Identity()
    else:
        raise NotImplementedError(
            'invalid activation function: {:s}'.format(actv)
        )

def make_norm2d(name, plane, affine=True):
    if name == 'batch':
        return nn.BatchNorm2d(plane, affine=affine)
    elif name == 'instance':
        return nn.InstanceNorm2d(plane, affine=affine)
    elif name == 'none':
        return nn.Identity()
    else:
        raise NotImplementedError(
            'invalid normalization function: {:s}'.format(name)
        )

def make_norm3d(name, plane, affine=True):
    if name == 'batch':
        return nn.BatchNorm3d(plane, affine=affine)
    elif name == 'instance':
        return nn.InstanceNorm3d(plane, affine=affine)
    elif name == 'max':
        return MaxNorm()
    elif name == 'none':
        return nn.Identity()
    else:
        raise NotImplementedError(
            'invalid normalization function: {:s}'.format(name)
        )


class MaxNorm(nn.Module):
    """ Per-channel normalization by max value """
    def __init__(self, eps=1e-8):
        super(MaxNorm, self).__init__()

        self.eps = eps

    def forward(self, x):
        """
        Args:
            x (float tensor, (bs, c, d, h, w)): raw RSD output.

        Returns:
            x (float tensor, (bs, c, d, h, w)): normalized RSD output.
        """
        assert x.dim() == 5, \
            'input should be a 5D tensor, got {:d}D'.format(x.dim())
        x = F.normalize(x, p=float('inf'), dim=(-3, -2, -1))
        return x


class Blur2d(nn.Module):
    """ 2D blur kernel """
    def __init__(self, in_plane):
        super(Blur2d, self).__init__()

        self.in_plane = in_plane

        weight = torch.Tensor([[1, 2, 1], [2, 4, 2], [1, 2, 1]]) / 16
        weight = weight.expand(in_plane, -1, -1).unsqueeze(1)
        self.register_buffer('weight', weight)        # (c, 1, 3, 3)

    def forward(self, x):
        x = F.pad(x, (1, 1, 1, 1), mode='reflect')
        x = F.conv2d(x, self.weight, groups=self.in_plane)
        return x


class ResConv2d(nn.Module):
    """ Residual block with 2D conv layers """
    def __init__(
        self, 
        in_plane,           # number of input planes.
        plane,              # number of intermediate and output planes
        stride=1,           # stride of first conv layer
        actv='leaky_relu',  # activation function
        norm='instance',    # normalization function
        affine=True,        # if True, apply learnable affine transform in norm
    ):
        super(ResConv2d, self).__init__()

        self.in_plane = in_plane
        self.plane = plane
        self.stride = stride
        bias = True if norm == 'none' or not affine else False

        self.conv1 = nn.Conv2d(
            in_plane, plane, 3, stride, 1, 
            padding_mode='replicate', bias=bias
        )
        self.norm1 = make_norm2d(norm, plane, affine)
        
        self.conv2 = nn.Conv2d(
            plane, plane, 3, 1, 1,
            padding_mode='replicate', bias=bias
        )
        self.norm2 = make_norm2d(norm, plane, affine)

        if stride > 1 or in_plane != plane:
            self.res_conv = nn.Conv2d(in_plane, plane, 1, stride, 0, bias=bias)
            self.res_norm = make_norm2d(norm, plane, affine)

        self.actv = make_actv(actv)

    def forward(self, x):
        dx = self.norm1(self.conv1(x))
        dx = self.actv(dx)
        dx = self.norm2(self.conv2(dx))
        if self.stride > 1 or self.in_plane != self.plane:
            x = self.res_norm(self.res_conv(x))
        x = self.actv(x + dx)
        return x


class ResConv3d(nn.Module):
    """ Residual block with 3D conv layers """
    def __init__(
        self, 
        in_plane,           # number of input planes
        plane,              # number of intermediate and output planes
        stride=1,           # stride of first conv layer
        actv='leaky_relu',  # activation function
        norm='none',        # normalization function
        affine=True,        # if True, apply learnable affine transform in norm
    ):
        super(ResConv3d, self).__init__()

        self.in_plane = in_plane
        self.plane = plane
        self.stride = stride
        bias = True if norm == 'none' or not affine else False

        self.conv1 = nn.Conv3d(
            in_plane, plane, 3, stride, 1, 
            padding_mode='replicate', bias=bias
        )
        self.norm1 = make_norm3d(norm, plane, affine)
        self.conv2 = nn.Conv3d(
            plane, plane, 3, 1, 1, 
            padding_mode='replicate', bias=bias
        )
        self.norm2 = make_norm3d(norm, plane, affine)

        if stride > 1 or in_plane != plane:
            self.res_conv = nn.Conv3d(in_plane, plane, 1, stride, 0, bias=bias)
            self.res_norm = make_norm3d(norm, plane, affine)

        self.actv = make_actv(actv)
    
    def forward(self, x):
        dx = self.norm1(self.conv1(x))
        dx = self.actv(dx)
        dx = self.norm2(self.conv2(dx))
        if self.stride > 1 or self.in_plane != self.plane:
            x = self.res_norm(self.res_conv(x))
        x = self.actv(x + dx)
        return x


class ResBlock2d(nn.Module):

    def __init__(
        self, 
        in_plane, 
        plane, 
        stride, 
        n_layers,  
        actv='relu', 
        norm='none', 
        affine=False,
    ):
        super(ResBlock2d, self).__init__()

        layers = []
        for i in range(n_layers):
            layers.append(
                ResConv2d(in_plane, plane, stride, actv, norm, affine)
            )
            in_plane = plane
            stride = 1
        self.layers = nn.Sequential(*layers)

        self.out_plane = in_plane

    def forward(self, x):
        x = self.layers(x)
        return x


class ResBlock3d(nn.Module):

    def __init__(
        self, 
        in_plane, 
        plane, 
        stride, 
        n_layers,  
        actv='relu', 
        norm='none', 
        affine=False,
    ):
        super(ResBlock3d, self).__init__()

        layers = []
        for i in range(n_layers):
            layers.append(
                ResConv3d(in_plane, plane, stride, actv, norm, affine)
            )
            in_plane = plane
            stride = 1
        self.layers = nn.Sequential(*layers)

        self.out_plane = in_plane

    def forward(self, x):
        x = self.layers(x)
        return x


class UpSample2d(nn.Module):
    """ 2D up-sampling layer """
    def __init__(
        self, 
        in_plane,           # number of input planes
        out_plane,          # number of output planes
        mode='bilinear',    # up-sampling method 
        refine='conv',      # refinement method following up-sampling
        actv='leaky_relu',  # activation function
        norm='instance',    # normalization function
        affine=True,        # if True, apply learnable affine transform in norm
    ):
        super(UpSample2d, self).__init__()
        assert mode in ('transpose', 'bilinear', 'nearest', 'shuffle'), \
            'invalid up-sampling method: {:s}'.format(mode)
        assert refine in ('none', 'conv', 'blur'), \
            'invalid refinement method: {:s}'.format(refine)

        bias = True if norm == 'none' or not affine else False

        if mode == 'transpose':
            block = nn.Sequential(
                nn.ConvTranspose2d(in_plane, out_plane, 4, 2, 1, bias=bias),
                make_norm2d(norm, out_plane, affine), 
                make_actv(actv),
            )
        elif mode == 'bilinear':
            block = nn.Upsample(
                scale_factor=2, mode='bilinear', align_corners=False
            )
            if refine == 'conv':
                block = nn.Sequential(
                    block,
                    nn.Conv2d(in_plane, out_plane, 3, 1, 1, bias=bias),
                    make_norm2d(norm, out_plane, affine),
                    make_actv(actv),
                )
            elif refine == 'blur':
                block = nn.Sequential(
                    block, 
                    Blur2d(in_plane),
                )
        elif mode == 'nearest':
            block = nn.Upsample(
                scale_factor=2, mode='nearest', align_corners=False
            )
            if refine == 'conv':
                block = nn.Sequential(
                    block,
                    nn.Conv2d(in_plane, out_plane, 3, 1, 1, bias=bias),
                    make_norm2d(norm, out_plane, affine),
                    make_actv(actv),
                )
            elif refine == 'blur':
                block = nn.Sequential(
                    block, 
                    Blur2d(in_plane),
                )
        else:
            block = nn.Sequential(
                nn.PixelShuffle(upscale_factor=2),
                nn.Conv2d(in_plane // 4, out_plane, 3, 1, 1, bias=bias),
                make_norm2d(norm, out_plane, affine),
                make_actv(actv),
            )

        self.block = block

    def forward(self, x):
        x = self.block(x)
        return x


class DownSample3d(nn.Module):
    """ 3D down-sampling layer """
    def __init__(
        self, 
        in_plane, 
        out_plane, 
        actv='leaky_relu', 
        norm='instance', 
        affine=True
    ):
        super(DownSample3d, self).__init__()

        bias = True if norm == 'none' or not affine else False

        self.block = nn.Sequential(
            nn.Conv3d(
                in_plane, out_plane, 4, 2, 1, 
                padding_mode='replicate', bias=bias
            ),
            make_norm3d(norm, out_plane, affine),
            make_actv(actv))

    def forward(self, x):
        x = self.block(x)
        return x


class UpSample3d(nn.Module):
    """ 3D up-sampling layer """
    def __init__(
        self, 
        in_plane, 
        out_plane,
        actv='relu', 
        norm='instance', 
        affine=True,
    ):
        super(UpSample3d, self).__init__()

        bias = True if norm == 'none' or not affine else False

        self.block = nn.Sequential(
            nn.ConvTranspose3d(in_plane, out_plane, 4, 2, 1, bias=bias),
            make_norm3d(norm, out_plane, affine),
            make_actv(actv))

    def forward(self, x):
        x = self.block(x)
        return x


class UNetBlock(nn.Module):
    """ UNet block (a.k.a. one level of UNet) """
    def __init__(
        self, 
        outer_plane, 
        inner_plane, 
        submodule=None, 
        skip=None,
        down_actv='leaky_relu', 
        up_actv='relu', 
        norm='instance', 
        affine=True,
    ):
        super(UNetBlock, self).__init__()

        if submodule is None:
            if skip is None:
                self.block = nn.Sequential(
                    DownSample3d(
                        in_plane=outer_plane, 
                        out_plane=inner_plane, 
                        actv=down_actv, 
                        norm=norm, 
                        affine=affine,
                    ),
                    UpSample3d(
                        in_plane=inner_plane, 
                        out_plane=outer_plane, 
                        actv=up_actv, 
                        norm=norm, 
                        affine=affine,
                    ),
                )
                self.skip = nn.Identity()
                self.out_plane = outer_plane * 2
            else:
                self.block = skip
                self.skip = None
                self.out_plane = outer_plane
        else:
            self.block = nn.Sequential(
                DownSample3d(
                    in_plane=outer_plane, 
                    out_plane=inner_plane, 
                    actv=down_actv, 
                    norm=norm, 
                    affine=affine,
                ),
                submodule,
                UpSample3d(
                    in_plane=submodule.out_plane, 
                    out_plane=outer_plane, 
                    actv=up_actv, 
                    norm=norm, 
                    affine=affine,
                ),
            )
            self.skip = skip if skip is not None else nn.Identity()
            self.out_plane = outer_plane * 2

    def forward(self, x):
        y = self.block(x)
        if self.skip is not None:
            y = torch.cat([self.skip(x), y], dim=1)
        return y