import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .modules import make_actv
from .clib import ray_aabb_intersect


class Linear(nn.Linear):
    """
    A FiLM-conditioned linear layer 
    (Perez et al., AAAI 18)
    """
    def __init__(self, in_dim, out_dim, init='none', actv='relu', w0=None):
        """
        Args:
            in_dim (int): input dimension.
            out_dim (int): output dimension.
            init (str): initialization method.
            actv (str): activation function.
            w0 (float): normalizing constant in weight initialization. 
                        This is used in the FiLM-conditioning layers for SIREN.
                        (NOTE: higher-frequency signals require larger w0.)
        """
        super(Linear, self).__init__(in_dim, out_dim)

        self.in_dim = in_dim
        self.out_dim = out_dim

        self.init_parameters(init, actv, w0)
        self.actv = make_actv(actv)

    def init_parameters(self, init, actv, w0=None):
        if init == 'none':
            return
        elif init == 'xavier':
            nn.init.xavier_uniform_(
                self.weight, gain=nn.init.calculate_gain(actv)
            )
            nn.init.zeros_(self.bias)
        elif init == 'kaiming':
            a = 0.2 if actv == 'leaky_relu' else 0
            nn.init.kaiming_normal_(
                self.weight, a=a, mode='fan_in', nonlinearity=actv
            )
            nn.init.zeros_(self.bias)
        elif init == 'zero':
            nn.init.zeros_(self.weight)
            nn.init.zeros_(self.bias)
        elif init == 'siren':
            assert w0 is not None, 'w0 must be specified for SIREN'
            bound = math.sqrt(6 / self.in_dim) / w0
            nn.init.uniform_(self.weight, -bound, bound)
        else:
            raise ValueError(
                'invalid initialization method: {:s}'.format(init)
            )

    def forward(self, x, gamma=None, beta=None, in_actv=False):
        if in_actv:
            x = self.actv(x)
        x = super().forward(x)
        
        # FiLM modulation
        if gamma is not None:
            assert gamma.shape == x.shape, 'gamma shape mismatch'
            x = gamma * x
        if beta is not None:
            assert beta.shape == x.shape, 'beta shape mismatch'
            x = x + beta
        
        if not in_actv:
            x = self.actv(x)
        return x


class SineLayer(nn.Module):
    """
    A FiLM-conditioned SIREN layer
    (Sitzmann et al., NeurIPS 20; Chan et al., arXiv 20)
    """
    def __init__(self, in_dim, out_dim, is_first=False, w0=30):
        """
        Args:
            in_dim (int): input dimension.
            out_dim (int): output dimension.
            is_first (bool): if True, initialize as first layer.
            w0 (float): normalizing constant in weight initialization.
                        (NOTE: higher-frequency signals require larger w0.)
        """
        super(SineLayer, self).__init__()

        self.in_dim = in_dim
        self.out_dim = out_dim
        self.is_first = is_first
        self.w0 = w0

        self.linear = nn.Linear(in_dim, out_dim)
        self.init_parameters()

    def init_parameters(self):
        bound = 1 / self.in_dim
        if self.is_first:
            nn.init.uniform_(self.linear.weight, -bound, bound)
        else:
            bound = math.sqrt(6 * bound) / self.w0
            nn.init.uniform_(self.linear.weight, -bound, bound)

    def forward(self, x, gamma=None, beta=None):
        x = self.w0 * self.linear(x)
        
        # FiLM modulation
        if gamma is not None:
            assert gamma.shape == x.shape, 'gamma shape mismatch'
            x = gamma * x
        if beta is not None:
            assert beta.shape == x.shape, 'beta shape mismatch'
            x = x + beta
        
        x = torch.sin(x)
        return x


class PosEmbedder(nn.Module):
    """
    Fourier feature embedding
    (Mildenhall et al., ECCV 20)
    """
    def __init__(
        self, 
        in_dim=3,           # input dimension
        include_input=True, # if True, include input in the embedding
        n_freqs=6,          # number of sinusoid frequencies
        log_sampling=True,  # if True, sample frequency band in log space
        freq_scale=1,       # frequency scale
    ):
        super(PosEmbedder, self).__init__()

        self.in_dim = in_dim
        self.include_input = include_input
        self.n_freqs = n_freqs
        self.log_sampling = log_sampling
        self.freq_scale = freq_scale

        max_freq = self.n_freqs - 1
        if self.log_sampling:
            freqs = 2 ** torch.linspace(0, max_freq, self.n_freqs)
        else:
            freqs = torch.linspace(1, 2 ** max_freq, self.n_freqs)
        self.register_buffer('freqs', freq_scale * freqs)

        self._create_embed_fns()

    def _create_embed_fns(self):
        fns = []
        d = self.in_dim
        out_dim = 0

        if self.include_input:
            fns.append(lambda x : x)
            out_dim += d 

        for freq in self.freqs:
            fns.append(lambda x, freq=freq : torch.sin(x * freq))
            fns.append(lambda x, freq=freq : torch.cos(x * freq))
            out_dim += 2 * d

        self.fns = fns
        self.out_dim = out_dim

    def forward(self, x):
        out = torch.cat([fn(x) for fn in self.fns], dim=-1)
        return out


class FiLMGenerator(nn.Module):
    """
    Affine parameter generator for FiLM conditioning 
    (Perez et al., AAAI 18; Chan et al., arXiv 20).
    """
    def __init__(
        self, 
        in_dim, 
        hid_dim, 
        out_dim, 
        n_layers=3, 
        actv='leaky_relu', 
        normalize=True,
    ):
        super(FiLMGenerator, self).__init__()

        self.in_dim = in_dim
        self.hid_dim = hid_dim
        self.out_dim = out_dim
        self.n_layers = n_layers
        self.normalize = normalize

        layers = []
        for _ in range(n_layers):
            layers.append(
                Linear(in_dim, hid_dim, init='kaiming', actv=actv)
            )
            in_dim = hid_dim
        self.layers = nn.ModuleList(layers)
        self.out_layer = Linear(
            in_dim, out_dim * 2, init='kaiming', actv='linear'
        )

    def forward(self, x):
        """
        Args:
            x (float tensor, (bs, c)): (embedded) latent code.

        Returns:
            gamma (float tensor, (bs, d)): affine weight.
            beta (float tensor, (bs, d)): affine bias.
        """
        if self.normalize:
            x = F.normalize(x, dim=-1)

        for i in range(self.n_layers):
            x = self.layers[i](x)
        out = self.out_layer(x)
        
        gamma, beta = out[:, :self.out_dim], out[:, self.out_dim:]
        return gamma, beta


class ResLinear(nn.Module):
    """
    Residual block of linear layers
    (Niemeyer et al., CVPR 20; Yu et al., arXiv 20).
    """
    def __init__(self, in_dim, hid_dim=None, out_dim=None, actv='relu'):
        """
        Args:
            in_dim (int): input dimension.
            hid_dim (int): hidden dimension.
            out_dim (int): output dimension.
            actv (str): activation function.
        """
        super(ResLinear, self).__init__()

        if out_dim is None:
            out_dim = in_dim
        if hid_dim is None:
            hid_dim = min(in_dim, out_dim)

        self.in_dim = in_dim
        self.hid_dim = hid_dim
        self.out_dim = out_dim

        self.fc1 = Linear(in_dim, hid_dim, init='kaiming', actv=actv)
        self.fc2 = Linear(hid_dim, out_dim, init='zero', actv=actv)

        if in_dim == out_dim:
            self.downsample = nn.Identity()
        else:
            self.downsample = Linear(
                in_dim, out_dim, init='kaiming', actv='linear'
            )

    def forward(self, x):
        dx = self.fc1(x, in_actv=True)
        dx = self.fc2(dx, in_actv=True)
        x = self.downsample(x) + dx
        return x


class ResNet(nn.Module):
    """
    A residual network parameterization of implicit volume function
    (Niemeyer et al., CVPR 20; Yu et al., arXiv 20)
    """
    def __init__(
        self, 
        p_dim,                  # size of (embedded) point coordinates
        d_dim,                  # size of (embedded) ray direction
        z_dim=None,             # size of (embedded) latent code
        hid_dim=256,            # number of hidden units
        color_dim=1,            # color dimension (1 for BW and 3 for RGB)
        n_blocks=5,             # number of residual blocks
        n_modulated_blocks=3,   # number of residual blocks modulated by latent code
        actv='relu',            # activation function
        use_spade=False,        # if True, predict affine weight in addition to bias
                                # from latent code to transform network activations
                                # (Park et al., CVPR 19)
    ):
        super(ResNet, self).__init__()

        self.p_dim = p_dim
        self.d_dim = d_dim
        self.z_dim = z_dim
        self.hid_dim = hid_dim
        self.color_dim = color_dim
        self.n_blocks = n_blocks
        self.n_modulated_blocks = n_modulated_blocks
        self.use_spade = use_spade

        self.fc_in = Linear(
            p_dim + d_dim, hid_dim, init='kaiming', actv='linear'
        )
        self.fc_out = Linear(
            hid_dim, 1 + color_dim, init='kaiming', actv=actv
        )
        self.blocks = nn.ModuleList(
            [ResLinear(hid_dim, actv=actv) for _ in range(n_blocks)]
        )

        if z_dim is not None:
            assert n_modulated_blocks <= n_blocks
            self.z_bias = nn.ModuleList(
                [Linear(z_dim, hid_dim, init='kaiming', actv='linear') \
                    for _ in range(n_modulated_blocks)]
            )
            if use_spade:
                self.z_scale = nn.ModuleList(
                    [Linear(z_dim, hid_dim, init='kaiming', actv='linear') \
                        for _ in range(n_modulated_blocks)]
                )

    def forward(self, p, d, z=None):
        """
        Args:
            p (float tensor, (bs, cx)): (embedded) point coordinates.
            d (float tensor, (bs, cd)): (embedded) ray direction.
            z (float tensor, (bs, cz)): (embedded) latent code.

        Returns:
            sigma (float tensor, (bs, 1)): volume density.
            color (float tensor, (bs, 1/3)): color.
        """
        x = self.fc_in(torch.cat([p, d], dim=-1))
        for i in range(self.n_blocks):
            if z is not None and i < self.n_modulated_blocks:
                bias = self.z_bias[i](z)
                if self.use_spade:
                    scale = self.z_scale[i](z)
                    x = scale * x
                x = x + bias
            x = self.blocks[i](x)
        out = self.fc_out(x, in_actv=True)
        
        sigma, color = out[:, :1], out[:, 1:]
        return sigma, color


class NSVF(nn.Module):
    """
    An NSVF parametrization of the implicit volume function
    (Liu et al., NeurIPS 20)
    """
    def __init__(
        self, 
        p_dim,              # size of (embedded) point coordinates
        d_dim,              # size of (embedded) ray direction
        z_dim=None,         # size of (embedded) latent code
        hid_dim=256,        # number of hidden units
        color_dim=1,        # color dimension (1 for BW and 3 for RGB)
        n_sigma_layers=4,   # number of layers for predicting volume density
        n_color_layers=4,   # number of layers for predicting color
        actv='relu',        # activation function
    ):
        super(NSVF, self).__init__()

        self.p_dim = p_dim
        self.d_dim = d_dim
        self.z_dim = z_dim
        self.hid_dim = hid_dim
        self.color_dim = color_dim
        self.n_sigma_layers = n_sigma_layers
        self.n_color_layers = n_color_layers

        in_dim = p_dim if z_dim is None else z_dim
        sigma_layers = []
        for i in range(n_sigma_layers):
            dim = hid_dim // 2 if i == n_sigma_layers - 1 else hid_dim
            sigma_layers.append(
                Linear(in_dim, dim, init='kaiming', actv=actv)
            )
            in_dim = dim
        self.sigma_layers = nn.ModuleList(sigma_layers)
        self.sigma_out = Linear(in_dim, 1, init='kaiming', actv='linear')

        in_dim = in_dim + d_dim if n_sigma_layers == 1 else hid_dim + d_dim
        color_layers = []
        for _ in range(n_color_layers):
            color_layers.append(
                Linear(in_dim, hid_dim, init='kaiming', actv=actv)
            )
            in_dim = hid_dim
        self.color_layers = nn.ModuleList(color_layers)
        self.color_out = Linear(
            in_dim, color_dim, init='kaiming', actv='linear'
        )

    def forward(self, p, d=None, z=None):
        """
        Args:
            p (float tensor, (bs, cx)): (embedded) point coordinates.
            d (float tensor, (bs, cd)): (embedded) ray directions.
            z (float tensor, [bs, cz)): (embedded) latent code.

        Returns:
            sigma (float tensor, (bs, 1)): volume density.
            color (float tensor, (bs, 1/3)): color.
        """
        x = p if z is None else z
        for i in range(self.n_sigma_layers - 1):
            x = self.sigma_layers[i](x)
        sigma = self.sigma_layers[-1](x)
        sigma = self.sigma_out(sigma)

        if d is None:
            return sigma, None

        x = torch.cat([x, d], -1)
        for i in range(self.n_color_layers):
            x = self.color_layers[i](x)
        color = self.color_out(x)

        return sigma, color


class SIREN(nn.Module):
    """
    A SIREN parametrization of implicit volume function
    (Sitzmann et al., NeurIPS 20)
    """
    def __init__(
        self, 
        p_dim,                  # size of (embedded) point coordinates
        d_dim,                  # size of (embedded) ray direction
        z_dim=None,             # size of (embedded) latent code
        hid_dim=256,            # number of hidden units
        color_dim=1,            # color dimension (1 for BW, 3 for RGB)
        n_sigma_layers=8,       # number of layers for predicting volume density
        n_color_layers=1,       # number of layers for predicting color
        in_w0=30,               # norm in weight init for first layer
        hid_w0=30,              # norm in weight init for hidden layers
        film_hid_dim=256,       # number of hidden units in FiLM
        film_n_layers=3,        # number of layers in FiLM
        film_actv='leaky_relu', # activation function in FiLM
        film_normalize=True,    # if True, normalize FiLM input
    ):
        super(SIREN, self).__init__()

        self.p_dim = p_dim
        self.d_dim = d_dim
        self.z_dim = z_dim
        self.hid_dim = hid_dim
        self.color_dim = color_dim
        self.n_sigma_layers = n_sigma_layers
        self.n_color_layers = n_color_layers
        self.n_layers = n_sigma_layers + n_color_layers

        self.film_generator = None
        if z_dim is not None:
            film_out_dim = self.n_layers * hid_dim
            self.film_generator = FiLMGenerator(
                in_dim=z_dim, 
                hid_dim=film_hid_dim, 
                out_dim=film_out_dim, 
                n_layers=film_n_layers, 
                actv=film_actv,
                normalize=film_normalize,
            )
        
        sigma_layers = []
        in_dim = p_dim
        for i in range(n_sigma_layers):
            sigma_layers.append(
                SineLayer(in_dim, hid_dim, i == 0, in_w0 if i == 0 else hid_w0)
            )
            in_dim = hid_dim
        self.sigma_layers = nn.ModuleList(sigma_layers)
        self.sigma_out = Linear(
            in_dim, 1, init='siren', actv='linear', w0=hid_w0
        )

        color_layers = []
        in_dim = hid_dim + d_dim
        for _ in range(n_color_layers):
            color_layers.append(
                SineLayer(in_dim, hid_dim, False, hid_w0)
            )
            in_dim = hid_dim
        self.color_layers = nn.ModuleList(color_layers)
        self.color_out = Linear(
            in_dim, color_dim, init='siren', actv='linear', w0=hid_w0
        )

    def forward(self, p, d=None, z=None):
        """
        Args:
            p (float tensor, (bs, cx)): (embedded) point coordinates.
            d (float tensor, (bs, cd)): (embedded) ray directions.
            z (float tensor, (bs, cz)): (embedded) latent code.

        Returns:
            sigma (float tensor, (bs, 1)): volume density.
            color (float tensor, (bs, 1/3)): color.
        """
        if z is not None:
            assert self.film_generator is not None, \
                'FiLM generator does not exist'
            gamma, beta = self.film_generator(z)
            gamma = torch.split(gamma, self.hid_dim, -1)
            beta = torch.split(beta, self.hid_dim, -1)
            assert len(gamma) == self.n_layers, 'gamma shape mismatch'
            assert len(beta) == self.n_layers, 'beta shape mismatch'
        else:
            gamma = beta = [None] * self.n_layers
        k = 0
        
        x = p
        for i in range(self.n_sigma_layers):
            x = self.sigma_layers[i](x, gamma[k], beta[k])
            k += 1
        sigma = self.sigma_out(x)

        if d is None:
            return sigma, None

        x = torch.cat([x, d], -1)
        for i in range(self.n_color_layers):
            x = self.color_layers[i](x, gamma[k], beta[k])
            k += 1
        color = self.color_out(x)

        return sigma, color


class NeRF(nn.Module):
    """
    A NeRF parametrization of implicit volume function
    (Mildenhall et al., ECCV 20)
    """
    def __init__(
        self, 
        p_dim,                  # size of (embedded) point coordinates 
        d_dim,                  # size of (embedded) ray direction
        z_dim=None,             # size of (embedded) latent code
        hid_dim=256,            # number of hidden units
        color_dim=1,            # color dimension (1 for BW, 3 for RGB)
        skips=[5],              # layers to add skip connection
        n_sigma_layers=8,       # number of layers for predicting volume density
        n_color_layers=1,       # number of layers for predicting color
        actv='relu',            # activation function
        film_hid_dim=256,       # number of hidden units in FiLM
        film_n_layers=3,        # number of layers in FiLM
        film_actv='leaky_relu', # activation function in FiLM
        film_normalize=True,    # if True, normalize FiLM input
    ):
        super(NeRF, self).__init__()

        self.p_dim = p_dim
        self.d_dim = d_dim
        self.z_dim = z_dim
        self.hid_dim = hid_dim
        self.color_dim = color_dim
        
        if isinstance(skips, int):
            skips = [skips]
        assert isinstance(skips, (list, tuple)), \
            'must specify skip layers as a list / tuple of one-based indices'
        self.skips = skips

        self.n_sigma_layers = n_sigma_layers
        self.n_color_layers = n_color_layers
        self.n_layers = n_sigma_layers + 1 + n_color_layers

        self.film_generator = None
        if z_dim is not None:
            film_out_dim = (n_sigma_layers + 1) * hid_dim \
                         + n_color_layers * hid_dim // 2
            self.split_size = [hid_dim] * (n_sigma_layers + 1) \
                            + [hid_dim // 2] * n_color_layers
            self.film_generator = FiLMGenerator(
                in_dim=z_dim, 
                hid_dim=film_hid_dim, 
                out_dim=film_out_dim, 
                n_layers=film_n_layers, 
                actv=film_actv,
                normalize=film_normalize,
            )
        
        sigma_layers = []
        in_dim = p_dim
        for i in range(1, n_sigma_layers + 1): 
            sigma_layers.append(
                Linear(in_dim, hid_dim, init='xavier', actv=actv)
            )
            in_dim = hid_dim if i + 1 not in skips else hid_dim + p_dim
        self.sigma_layers = nn.ModuleList(sigma_layers)
        self.sigma_out = Linear(in_dim, 1, init='xavier', actv='linear')

        self.feat_layer = Linear(
            in_dim, hid_dim, init='xavier', actv='linear'
        )

        in_dim = hid_dim + d_dim
        color_layers = []
        for _ in range(n_color_layers):
            color_layers.append(
                Linear(in_dim, hid_dim // 2, init='xavier', actv=actv)
            )
            in_dim = hid_dim // 2
        self.color_layers = nn.ModuleList(color_layers)
        self.color_out = Linear(
            in_dim, color_dim, init='xavier', actv='linear'
        )

    def forward(self, p, d=None, z=None):
        """
        Args:
            p (float tensor, (bs, cx)): (embedded) point coordinates.
            d (float tensor, (bs, cd)): (embedded) ray directions.
            z (float tensor, (bs, cz)): (embedded) latent code.

        Returns:
            sigma (float tensor, (bs, 1)): volume density.
            color (float tensor, (bs, 1/3)): color.
        """
        if z is not None:
            assert self.film_generator is not None, \
                'FiLM generator does not exist'
            gamma, beta = self.film_generator(z)
            gamma = torch.split(gamma, self.split_size, -1)
            beta = torch.split(beta, self.split_size, -1)
            assert len(gamma) == self.n_layers, 'gamma shape mismatch'
            assert len(beta) == self.n_layers, 'beta shape mismatch'
        else:
            gamma = beta = [None] * self.n_layers
        k = 0

        x = p
        for i in range(self.n_sigma_layers):
            if i + 1 in self.skips:
                x = torch.cat([x, p], -1)
            x = self.sigma_layers[i](x, gamma[k], beta[k])
            k += 1
        sigma = self.sigma_out(x)
        
        if d is None:
            return sigma, None

        x = self.feat_layer(x, gamma[k], beta[k])
        k += 1

        x = torch.cat([x, d], -1)
        for i in range(self.n_color_layers):
            x = self.color_layers[i](x, gamma[k], beta[k])
            k += 1
        color = self.color_out(x)

        return sigma, color


class Renderer(nn.Module):
    """
    A renderer backbone for transient / steady-state rendering
    (NOTE: this implementation assumes LEFT-handed system)
    """
    def __init__(
        self, 
        p_embed_fn,             # embedding function for point coordinates
        d_embed_fn,             # embedding function for ray direction
        z_embed_fn,             # embedding function for latent code
        field_fn,               # implicit volume function
        bb_ctr=[0, 0, 0],       # bounding box center w.r.t. world frame (unit: m)
        bb_size=[2, 2, 2],      # bounding box size (unit: m)
        inf=10,                 # depth at infinity (unit: m)
        p_polar=False,          # if True, represent point coordinates in polar system 
        d_polar=True,           # if True, represent ray direction in polar system
        z_norm=False,           # if True, normalize latent code
        sigma_transform='relu', # output transform for volume density prediction
        color_transform='relu', # output transform for color prediction
        learn_scale=False,      # if True, learn a scale factor
    ):
        super(Renderer, self).__init__()

        self.p_embed_fn = p_embed_fn
        self.d_embed_fn = d_embed_fn
        self.z_embed_fn = z_embed_fn
        self.field_fn = field_fn

        self.p_polar = p_polar
        self.d_polar = d_polar
        self.z_norm = z_norm

        self.sigma_transform = make_actv(sigma_transform)
        self.color_transform = make_actv(color_transform)

        self.scale = None
        if learn_scale:
            self.scale = nn.Parameter(torch.zeros(1))

        # set up the rendering volume
        assert isinstance(bb_ctr, (list, tuple)) and len(bb_ctr) == 3
        self.register_buffer('bb_ctr', torch.tensor(bb_ctr).float())
        if isinstance(bb_size, (int, float)):
            bb_size = [bb_size] * 3
        assert isinstance(bb_size, (list, tuple)) and len(bb_size) == 3
        self.register_buffer('bb_size', torch.tensor(bb_size).float())

        self.near = bb_ctr[-1] - bb_size[-1] / 2    # depth of near plane
        self.far = bb_ctr[-1] + bb_size[-1] / 2     # depth of far plane
        self.inf = inf                              # depth at infinity
        
    def sample_steps(self, o, d, n_steps):
        raise NotImplementedError(
            'must implement a mechanism for sampling ray-marching steps'
        )

    def sample_features(self, z_vol, p):
        """
        Sample a volumetric feature grid at given coordinates.

        Args:
            z_vol (float tensor, (bs, c, d, h, w)): feature volumes.
            p (float tensor, ((bs,) s, 3)): normalized xyz-coordinates of points.
                Each component of the coordinates lies in [-1, 1].

        Returns:
            z (float tensor, (bs, s, c)): sampled features.

        (bs: number of samples, 
         s: number of query points per sample,
        )
        """
        bs, s = z_vol.size(0), p.size(-2)
        if p.dim() == 2:
            p = p.repeat(bs, 1, 1)
        assert p.size(0) == bs, 'batch size mismatch'
        p = p.reshape(bs, s, 1, 1, 3)

        # trilinear interpolation
        ## NOTE: F.grid_sample assumes that y-axis points DOWNWARD
        p[..., 1] = -p[..., 1]  # flip y-axis
        z = F.grid_sample(z_vol, p, align_corners=False)   # (bs, c, s, 1, 1)
        z = z[..., 0, 0]                                   # (bs, c, s)
        z = z.permute(0, 2, 1)                             # (bs, s, c)
        return z

    def cartesian2polar(self, x, scale=False):
        # normalize input to unit vector
        r = torch.norm(x, dim=-1, keepdim=True).add_(1e-8)
        x.div_(r).clamp_(-1, 1)

        # unit [x, y, z] -> [theta, phi]
        theta = x[:, 2].acos_()
        phi = x[:, 1].div_(x[:, 0].add_(1e-8)).atan_()

        # normalize polar coordinates to [-1, 1]
        theta.sub_(math.pi / 2).div_(math.pi / 2)
        phi.div_(math.pi / 2)

        x = torch.stack([phi, theta], dim=-1)
        if scale:
            x = torch.cat([x, r], dim=-1)
        return x

    def evaluate_field(
            self, 
            p,
            d=None,
            z_vol=None,
            transient=False,
            sigma_noise=0,
            color_noise=0,
        ):
        """
        Evaluate neural radiance field.

        Args:
            p (float tensor, (s, 3)): raw xyz-coordinates.
            d (float tensor, (s, 3)): raw unit viewing directions.
            z_vol (float tensor, (bs, c, d, h, w)): raw feature volumes.
            transient (bool): if True, performs transient rendering.
            sigma_noise (float): noise added to raw sigma values.
            color_noise (float): noise added to raw radiance values.

        Returns:
            sigma (float tensor, (bs*s, 1)): rectified sigma values.
            color (float tensor, (bs*s, 1/3)): rectified color values.

        (s: total number of query points per sample)
        """
        # normalize xyz-coordinates to [-1, 1]
        ## NOTE: y-axis points UPWARD
        norm_p = (p - self.bb_ctr).div_(self.bb_size / 2)

        bs = 1
        # sample feature volume
        z = None
        if z_vol is not None:
            bs = z_vol.size(0)
            z = self.sample_features(z_vol, norm_p)             # (bs, s, c)
            if self.z_norm:
                z = F.normalize(z, dim=-1)
            z = z.flatten(0, 1)                                 # (bs*s, c)
            if self.z_embed_fn is not None:
                z = self.z_embed_fn(z)                          # (bs*s, cz)
        
        # embed spatial points
        if self.p_polar:
            p = self.cartesian2polar(p, scale=True)
        else:
            p = norm_p
        if self.p_embed_fn is not None:
            p = self.p_embed_fn(p)                              # (s, cx)
        p = p.repeat(bs, 1)                                     # (bs*s, cx)
        
        # embed ray directions
        if d is not None:
            d.mul_(-1)    # flip ray directions
            if self.d_polar:
                d = self.cartesian2polar(d)
            if self.d_embed_fn is not None:
                d = self.d_embed_fn(d)                          # (s, cd)
            d = d.repeat(bs, 1)                                 # (bs*s, cd)
        
        sigma, color = self.field_fn(p, d, z)                   # (bs*s, 1/3)

        # rectify sigma values
        if sigma_noise > 0:
            sigma_noise = torch.randn_like(sigma).mul_(sigma_noise)
            sigma = sigma + sigma_noise
        sigma = self.sigma_transform(sigma)                     # (bs*s, 1)

        # rectify color values
        if color is not None:
            if color_noise > 0:
                color_noise = torch.randn_like(color).mul_(color_noise)
                color = color + color_noise
            color = self.color_transform(color)                 # (bs*s, 1/3)

        return sigma, color

    def evaluate_irradiance(self, color, weight, lib):
        raise NotImplementedError(
            'must implement a mechanism to collect irradiance values'
        )

    def forward(
        self, 
        rays, 
        z_vol, 
        n_steps, 
        scale, 
        sigma_noise, 
        color_noise,
    ):
        """
        Args:
            rays (float tensor, (v, r, 7)): ray (origin, direction, weight) bundles.
            z_vol (float tensor, (bs, c, d, h, w)): feature volumes.
            n_steps (int): number of ray-marching steps PER BIN.
            sigma_noise (float): noise added to raw sigma values.
            color_noise (float): noise added to raw radiance values.

        Returns:
            lib (dict): intermediate quantities from rendering.

        (v : number of views, 
         r : number of rays per view,
         bs: number of samples, 
         c : feature dimension, 
         d : depth, h: height, w: width,
        )
        """
        if rays.dim() == 4:
            rays = rays[0]
        assert rays.dim() == 3
        assert rays.size(-1) == 7, 'ray bundle size mismatch'

        bs = z_vol.size(0) if z_vol is not None else 1
        v, r = rays.shape[:2]
        n = v * r               # number of rays PER SAMPLE

        o = rays[..., :3].reshape(n, 3)        # ray origins            # (n, 3)
        d = rays[..., 3:6].reshape(n, 3)       # ray directions         # (n, 3)
        w = rays[..., 6:]                      # ray weights            # (v, r, 1)

        # sample ray-marching steps
        steps, deltas, valid = self.sample_steps(o, d, n_steps)         # (n, m)
        m = steps.size(-1)      # number of SAMPLED steps PER RAY
        q = valid.sum(-1)       # number of VALID steps PER RAY         # (n,)

        depths = steps.masked_select(valid)                             # (s,)
        deltas = deltas.masked_select(valid)                            # (s,)
        
        valid = valid.repeat(bs, 1)                                     # (bs*n, m)
        deltas = deltas.repeat(bs)                                      # (bs*s,)

        # evaluate xyz-coordinates of query points
        o = o.repeat_interleave(q, 0)                                   # (s, 3)
        d = d.repeat_interleave(q, 0)                                   # (s, 3)
        p = o.add_(depths.unsqueeze(-1) * d)                            # (s, 3)

        # evaluate radiance field
        sigma, color = \
            self.evaluate_field(p, d, z_vol, sigma_noise, color_noise)  # (bs*s, 1/3)
        c = color.size(-1)      # number of color channels
        sigma = sigma.flatten()                                         # (bs*s,)
        color = color.flatten()                                         # (bs*s(*3),)

        # evaluate transmittance (t) and opacity (alpha)
        log_t = -sigma * deltas                                         # (bs*s,)
        alpha = 1 - torch.exp(log_t)                                    # (bs*s,)

        sigma = sigma.new_zeros(bs * n, m).masked_scatter_(valid, sigma)
        log_t = log_t.new_zeros(bs * n, m).masked_scatter_(valid, log_t)
        alpha = alpha.new_zeros(bs * n, m).masked_scatter_(valid, alpha)
        sigma = sigma.reshape(bs, n, m)                                 # (bs, n, m)
        log_t = log_t.reshape(bs, n, m)                                 # (bs, n, m)
        alpha = alpha.reshape(bs, n, m)                                 # (bs, n, m)

        valid = valid.repeat(1, c)
        color = color.new_zeros(bs * n, m * c).masked_scatter_(valid, color)
        color = color.reshape(bs, n, m, c)                              # (bs, n, m, 1/3)

        # evaluate cumulative transmittance and radiance
        clog_t = torch.cumsum(
            torch.cat([log_t.new_zeros(bs, n, 1), log_t], dim=-1), 
            dim=-1,
        )                                                               # (bs, n, m+1)
        weight = alpha * torch.exp(clog_t[..., :-1])                    # (bs, n, m)
        color = weight.unsqueeze(-1) * color                            # (bs, n, m, 1/3)
        
        # evaluate depth map and collision probability
        steps = steps.reshape(n, m)
        hit = weight.sum(-1) + 1e-10                                    # (bs, n)
        dist = (torch.sum(weight * steps, -1) + 1e-3) / hit             # (bs, n)
        dist = torch.min(torch.ones_like(dist) * self.inf, dist)        # (bs, n)

        lib = {
            'color': color.reshape(bs, v, r, m, c),    # piecewise radiance
            'sigma': sigma.reshape(bs, v, r, m),       # piecewise density
            'alpha': alpha.reshape(bs, v, r, m),       # piecewise opacity
            'log_t': log_t.reshape(bs, v, r, m),       # piecewise log-transmittance
            'clog_t': clog_t.reshape(bs, v, r, m + 1), # cumulative log-transmittance
            'weight': weight.reshape(bs, v, r, m),     # piecewise hit probability
            'hit': hit.reshape(bs, v, r, 1),           # per-ray hit probability
            'dist': dist.reshape(bs, v, r),            # per-ray expected depth
        }

        # evaluate irradiance
        lib['render'] = self.evaluate_irradiance(lib['color'], w, lib) * scale
        if self.scale is not None:
            lib['render'] = torch.exp(self.scale) * lib['render']

        return lib
        

class TransientRenderer(Renderer):
    """
    Transient volume renderer
    (NOTE: this implementation assumes LEFT-handed system)
    """
    def __init__(
        self, 
        p_embed_fn,             # embedding function for point coordinates
        d_embed_fn,             # embedding function for ray direction
        z_embed_fn,             # embedding function for latent code
        field_fn,               # implicit volume function
        bb_ctr=[0, 0, 0],       # bounding box center w.r.t. world frame
        bb_size=[2, 2, 2],      # bounding box size
        inf=10,                 # depth at infinity
        light=[0, 0, -1],       # light position w.r.t. world frame
        bin_range=[50, 150],    # histogram bin range for rendering
        bin_len=0.03,           # distance covered by a bin
        p_polar=False,          # if True, represent point coordinates in polar system
        d_polar=True,           # if True, represent ray direction in polar system
        z_norm=False,           # if True, normalize latent code
        sigma_transform='relu', # output transform for predicted volume density
        color_transform='relu', # output transform for predicted color
        learn_scale=False,      # if True, learn a scale factor
    ):
        super(TransientRenderer, self).__init__(
            p_embed_fn=p_embed_fn, 
            d_embed_fn=d_embed_fn, 
            z_embed_fn=z_embed_fn, 
            field_fn=field_fn, 
            bb_ctr=bb_ctr, 
            bb_size=bb_size, 
            inf=inf,
            p_polar=p_polar, 
            d_polar=d_polar, 
            sigma_transform=sigma_transform, 
            color_transform=color_transform,
            learn_scale=learn_scale,
        )

        assert isinstance(bin_range, (list, tuple)) and len(bin_range) == 2, \
            'must specify the first and the last indices of relavent bins'
        self.bin_range = bin_range
        self.n_bins = bin_range[1] - bin_range[0]
        self.bin_res = bin_len / 3e8
        
        self.t0 = bin_range[0] * self.bin_res  # traveling time for shortest path
        self.t1 = bin_range[1] * self.bin_res  # traveling time for longest path
        
        assert isinstance(light, (list, tuple)) and len(light) == 3
        self.register_buffer('light', torch.tensor(light).float())
        self.register_buffer(
            'tics', torch.linspace(self.t0, self.t1, self.n_bins + 1)
        )   # (b+1,)

    def sample_steps(self, o, d, n_steps=1):
        """
        Samples light traveling time uniformly at random. This ensures that every 
        bin is covered by the same number of samples. Note that the samples are NOT
        uniformly distributed along a ray (unless light and sensor are co-located).

        Args:
            o (float tensor, (n, 3)): ray origins (x, y, z).
            d (float tensor, (n, 3)): ray unit directions (x, y, z).
            n_steps (int): number of steps PER BIN.

        Returns:
            steps (float tensor, (n, bs*s)): sampled ray-marching steps.
            deltas (float tensor, (n, bs*s)): step sizes.
            valid (bool tensor, (n, bs*s)): a mask that indicates valid indices.

        (n: number of rays, 
         b: number of bins, 
         s: number of steps per bin,
        )
        """
        n, b = o.size(0), self.n_bins
        tics = self.tics.repeat(n, 1)
        left = tics[:, :-1].unsqueeze(-1)    # left bin edges           # (n, b, 1)
        right = tics[:, -1:]                 # right bin edges          # (n, 1)
        
        # stratified sampling of light traveling time
        eta = torch.rand((n, b, max(1, n_steps - 1)), device=o.device)  # (n, b, s-1)
        sub_res = self.bin_res / max(1, n_steps - 1)
        sub_inc = eta.new_ones(n, b, max(0, n_steps - 2)).mul_(sub_res)
        sub_left = torch.cumsum(torch.cat([left, sub_inc], -1), -1)     # (n, b, s-1)
        t = eta.mul_(sub_res).add_(sub_left)
        t = torch.cat([left, t], -1)                                    # (n, b, s)
        t = torch.cat([t.reshape(n, -1), right], -1)                    # (n, b*s+1)
        path_len = t.mul_(3e8)                                          # (n, b*s+1)

        # calculate ray-marching steps
        l = self.light - o                                              # (n, 3)
        l2 = torch.sum(l ** 2, -1, keepdim=True)                        # (n, 1)
        tmp1 = (path_len ** 2).sub_(l2)                                 # (n, b*s+1)
        tmp2 = torch.sum(l.mul_(d), -1, keepdim=True)                   # (n, 1)
        steps = tmp1 / path_len.sub_(tmp2).mul_(2)                      # (n, b*s+1)
        deltas = steps[:, 1:] - steps[:, :-1]                           # (n, b*s)
        if n_steps == 1:
            deltas = deltas.reshape(n, b, -1).sum(-1)                   # (n, b)
            steps = steps[:, 1::2]                                      # (n, b)
            tmp1 = tmp1[:, 1::2]                                        # (n, b)
        else:
            steps = steps[:, :-1].add_(deltas / 2)  # mid-point rule    # (n, b*s)
            tmp1 = tmp1[:, :-1]                                         # (n, b*s)

        # mark valid steps
        ## a. path length must be at least l
        ## b. samples must lie inside the volume
        near, far = ray_aabb_intersect(o, d, self.bb_ctr, self.bb_size) # (n, 1)
        valid = (tmp1 > 0) * (steps > near) * (steps < far)             # (n, b*s)
        
        return steps, deltas, valid

    def evaluate_irradiance(self, color, weight, lib=None):
        """
        Evaluate the weighted contribution of all rays leaving the same sensor.

        Args:
            color (float tensor, (bs, v, r, m, 1/3)): per-ray per-bin radiance.
            weight (float tensor, ((bs/1,) v, r, 1)): per-ray importance weights.

        Returns:
            irradiance (floart tensor, (bs, v, t, 1/3)): per-bin irradiance.

        (bs: batch size, 
         v : number of views, 
         r : number of rays per view,
         m : number of samples per ray, 
         t : number of temporal bins,
        )
        """
        color = color.reshape(*color.shape[:3], self.n_bins, -1, color.shape[-1])
        color = color.sum(-2)
        irradiance = torch.sum(color * weight.unsqueeze(-1), -3)
        return irradiance


class SteadyStateRenderer(Renderer):
    """
    Steady-state volume renderer
    (NOTE: this implementation assumes LEFT-handed system)
    """
    def __init__(
        self, 
        p_embed_fn,             # embedding function for point coordinates
        d_embed_fn,             # embedding function for ray direction
        z_embed_fn,             # embedding function for latent code
        field_fn,               # implicit volume function
        bb_ctr=[0, 0, 0],       # bounding box center w.r.t. world frame
        bb_size=[2, 2, 2],      # bounding box size
        inf=10,                 # depth at infinity
        bin_len=0.03,           # distance covered by a bin
        n_bins=64,              # number of bins
        p_polar=False,          # if True, represent point coordinates in polar system
        d_polar=True,           # if True, represent ray direction in polar system
        z_norm=False,           # if True, normalize latent code
        sigma_transform='relu', # output transform for predicted volume density
        color_transform='relu', # output transform for predicted color
        learn_scale=False,      # if True, learn a scale factor
        white_background=False, # if True, render with white background
    ):
        super(SteadyStateRenderer, self).__init__(
            p_embed_fn=p_embed_fn, 
            d_embed_fn=d_embed_fn, 
            z_embed_fn=z_embed_fn, 
            field_fn=field_fn, 
            bb_ctr=bb_ctr, 
            bb_size=bb_size, 
            inf=inf,
            p_polar=p_polar, 
            d_polar=d_polar, 
            sigma_transform=sigma_transform, 
            color_transform=color_transform,
            learn_scale=learn_scale,
        )

        self.bin_len = bin_len
        self.n_bins = n_bins
        self.white_background = white_background

    def sample_steps(self, o, d, n_steps=1):
        """
        Samples ray-marching steps uniformly at random. This ensures a good 
        Riemann-sum approximatation to the transmittance along a ray.

        Args:
            o (float tensor, (n, 3)): ray origins (x, y, z).
            d (float tensor, (n, 3)): ray unit directions (x, y, z).
            n_steps (int): number of steps PER BIN.
            prob (float tensor, (n, b)): 1D histograms as sampling distributions.
                if None, do uniform stratified sampling.

        Returns:
            steps (float tensor, (n, b*s)): sampled ray-marching steps.
            deltas (float tensor, (n, b*s)): step sizes.
            valid (bool tensor, (n, b*s)): a mask that indicates valid indices.

        (n: number of rays, 
         b: number of bins, 
         s: number of steps per bin,
         q: number of steps per ray,
        )
        """
        n, b = o.size(0), self.n_bins
        near, far = ray_aabb_intersect(o, d, self.bb_ctr, self.bb_size) # (n, 1)

        # ignore rays that do not intersect with the volume
        ## NOTE: near = far = -1 if no intersection
        mask = near.flatten() > 0                                       # (n,)
        o, d = o[mask], d[mask]                                         # (m, 3)
        near, far = near[mask], far[mask]                               # (m, 1)
        
        m = o.size(0)       # total number of VALID rays
        deltas = self.bin_len * near.new_ones(m, b)
        tics = torch.cumsum(torch.cat([near, deltas], -1), -1)          # (m, b+1)
        left = tics[:, :-1].unsqueeze(-1)                               # (m, b, 1)
        right = tics[:, -1:]                                            # (m, 1)

        eta = torch.rand((m, b, max(1, n_steps - 1)), device=o.device)
        sub_len = self.bin_len / max(1, n_steps - 1)
        sub_inc = eta.new_ones(m, b, max(0, n_steps - 2)).mul_(sub_len)
        sub_left = torch.cumsum(torch.cat([left, sub_inc], -1), -1)
        steps = eta.mul_(sub_len).add_(sub_left)
        if n_steps == 1:
            steps = steps.reshape(m, -1)                                # (m, b)
        else:
            steps = torch.cat([left, steps], -1)
            steps = torch.cat([steps.reshape(m, -1), right], -1)
            deltas = steps[:, 1:] - steps[:, :-1]                       # (m, b*s)
            steps = steps[:, :-1].add_(deltas / 2)                      # (m, b*s)
        
        # mark valid steps
        ## a. rays must intersect with the volume
        ## b. samples must lie inside the volume
        steps[steps > far] = -1
        tmp1 = -steps.new_ones(n, n_steps * b)
        tmp2 = deltas.new_zeros(n, n_steps * b)
        tmp1[mask] = steps
        tmp2[mask] = deltas
        steps, deltas = tmp1, tmp2                                      # (n, b*s)
        valid = steps > 0                                               # (n, b*s)
        
        return steps, deltas, valid

    def evaluate_irradiance(self, color, weight=None, lib=None):
        """
        Evaluate the sum of radiance along each ray.

        Args:
            color (float tensor, (bs, v, r, m, 1/3)): per-ray per-bin radiance.
            weight (float tensor, ((bs/1,) v, r, 1)): per-ray importance weights.

        Returns:
            irradiance (float tensor, (bs, v, r, 1/3)): per-ray irradiance.

        (bs: number of samples, 
         v: number of views, 
         r: number of rays per view,
         m: number of samples per ray,
        )
        """
        if weight is None:
            irradiance = torch.sum(color, -2)
        else:
            irradiance = torch.sum(color * weight.unsqueeze(-1), -2)

        if self.white_background:
            irradiance = irradiance + (1 - lib['hit'])
        return irradiance


def make_renderer(config):
    if config is None:
        return None, None
        
    # make positional embedder
    cf = config['embedder']
    
    if cf['embed_p']:
        p_embed_fn = PosEmbedder(**cf['p'])
        p_dim = p_embed_fn.out_dim
    else:
        p_embed_fn = None
        p_dim = cf['p']['in_dim']
    
    if cf['embed_d']:
        d_embed_fn = PosEmbedder(**cf['d'])
        d_dim = d_embed_fn.out_dim
    else:
        d_embed_fn = None
        d_dim = cf['d']['in_dim']

    if cf['embed_z']:
        z_embed_fn = PosEmbedder(**cf['z'])
        z_dim = z_embed_fn.out_dim
    else:
        z_embed_fn = None
        z_dim = cf['z']['in_dim']

    # make radiance field
    cf = config['field']

    if cf['type'] == 'nerf':
        cf['nerf']['p_dim'] = p_dim
        cf['nerf']['d_dim'] = d_dim
        cf['nerf']['z_dim'] = z_dim
        field_fn = NeRF(**cf['nerf'])
    
    elif cf['type'] == 'nsvf':
        cf['nsvf']['p_dim'] = p_dim
        cf['nsvf']['d_dim'] = d_dim
        cf['nsvf']['z_dim'] = z_dim
        field_fn = NSVF(**cf['nsvf'])
    
    elif cf['type'] == 'resnet':
        cf['resnet']['p_dim'] = p_dim
        cf['resnet']['d_dim'] = d_dim
        cf['resnet']['z_dim'] = z_dim
        field_fn = ResNet(**cf['resnet'])
    
    elif cf['type'] == 'siren':
        cf['siren']['p_dim'] = p_dim
        cf['siren']['d_dim'] = d_dim
        cf['siren']['z_dim'] = z_dim
        field_fn = SIREN(**cf['siren'])
    
    else:
        raise NotImplementedError(
            'invalid implicit volume function: {:s}'.format(cf['type'])
        )

    t_renderer = None
    if config.get('transient') is not None:
        t_renderer = TransientRenderer(
            p_embed_fn, d_embed_fn, z_embed_fn, field_fn, 
            **config['common'], **config['transient'])
        if torch.cuda.is_available():
            t_renderer.cuda()

    s_renderer = None
    if config.get('steady_state') is not None:
        s_renderer = SteadyStateRenderer(
            p_embed_fn, d_embed_fn, z_embed_fn, field_fn,
            **config['common'], **config['steady_state'])
        if torch.cuda.is_available():
            s_renderer.cuda()

    return t_renderer, s_renderer