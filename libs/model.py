import torch
import torch.nn as nn
import torch.utils.checkpoint as cp

from .modules import MaxNorm
from .encoder import make_encoder
from .renderer import make_renderer
from .decoder import make_decoder


def make_renderer_model(config):
    t_renderer, s_renderer = make_renderer(config)
    model = RendererModel(t_renderer, s_renderer)
    if torch.cuda.is_available():
        model.cuda()
    return model

def make_encoder_renderer_model(config):
    encoder = make_encoder(config['encoder'])
    t_renderer, s_renderer = make_renderer(config['renderer'])
    model = EncoderRendererModel(encoder, t_renderer, s_renderer)
    if torch.cuda.is_available():
        model.cuda()
    return model

def make_encoder_decoder_model(config):
    encoder = make_encoder(config['encoder'])
    decoder = make_decoder(config['decoder'])
    model = EncoderDecoderModel(encoder, decoder)
    if torch.cuda.is_available():
        model = model.cuda()
    return model


class RendererModel(nn.Module):
    """ Renderer model for per-scene optimization. """
    def __init__(self, t_renderer=None, s_renderer=None):
        super(RendererModel, self).__init__()
        assert t_renderer is not None or s_renderer is not None, \
            'at least one of the renderers must be specified'

        self.t_renderer = t_renderer    # transient renderer
        self.s_renderer = s_renderer    # steady-state renderer

    def forward(
        self, 
        wall_rays, 
        cam_rays, 
        n_steps,
        t_scale=1, 
        s_scale=1, 
        sigma_noise=0, 
        color_noise=0,
    ):
        """
        Args:
            wall_rays (float tensor, (1, vt, rt, 7)): rays for transient rendering.
            cam_rays (float tensor, (1, vs, rs, 7)): rays for steady-state rendering.
            n_steps (int): number of steps PER BIN.
            t_scale (float): fixed scaling factor for histograms.
            s_scale (float): fixed scaling factor for images.
            sigma_noise (float): noise added to raw sigma values.
            color_noise (float): noise added to raw radiance values.

        Returns:
            t_lib (dict): output from transient renderer.
            s_lib (dict): output from steady-state renderer.
        """
        t_lib = None
        if wall_rays is not None:
            assert self.t_renderer is not None, \
                'transient renderer doe not exist'
            t_lib = self.t_renderer(
                rays=wall_rays, 
                z_vol=None, 
                n_steps=n_steps, 
                scale=t_scale,
                sigma_noise=sigma_noise, 
                color_noise=color_noise,
            )

        s_lib = None
        if cam_rays is not None:
            assert self.s_renderer is not None, \
                'steady-state renderer does not exist'
            s_lib = self.s_renderer(
                rays=cam_rays, 
                z_vol=None, 
                n_steps=n_steps, 
                scale=s_scale,
                sigma_noise=sigma_noise, 
                color_noise=color_noise,
            )

        return t_lib, s_lib


class EncoderRendererModel(nn.Module):
    """ Encoder-renderer model for feed-forward inference """
    def __init__(self, encoder, t_renderer=None, s_renderer=None):
        super(EncoderRendererModel, self).__init__()
        assert t_renderer is not None or s_renderer is not None, \
            'at least one of the renderers must be specified'
        
        self.encoder = encoder
        self.t_renderer = t_renderer    # transient renderer
        self.s_renderer = s_renderer    # steady-state renderer

        self.norm = MaxNorm(per_channel=False)

    def _run_forward(self, module):
        def custom_forward(*inputs):
            inputs = module(inputs[0])
            return inputs
        return custom_forward

    def forward(
        self, 
        meas, 
        wall_rays, 
        cam_rays, 
        n_steps, 
        in_scale=None, 
        t_scale=1, 
        s_scale=1, 
        sigma_noise=0, 
        color_noise=0, 
        ckpt=False,
    ):
        """
        Args:
            meas (float tensor, (bs, 1, t, h, w)): transient measurements.
            wall_rays (float tensor, (1, vt, rt, 7)): rays for transient rendering.
            cam_rays (float tensor, (1, vs, rs, 7)): rays for steady-state rendering.
            n_steps (int): number of steps PER BIN.
            in_scale (float): fixed scaling factor for input.
            t_scale (float): fixed scaling factor for histograms.
            s_scale (float): fixed scaling factor for brightness images.
            sigma_noise (float): noise added to raw sigma values.
            color_noise (float): noise added to raw radiance values.

        Returns:
            t_lib (dict): output from transient renderer.
            s_lib (dict): output from steady-state renderer.
            feat_vol (float tensor, (bs, c, d, h, w)): feature volume.
        """
        if in_scale is None:
            meas = self.norm(meas)
        else:
            meas = meas * in_scale

        if ckpt:
            meas.requires_grad = True
            feat_vol = cp.checkpoint(self._run_forward(self.encoder), meas)
        else:
            feat_vol = self.encoder(meas)

        if isinstance(n_steps, (list, tuple)):
            t_steps, s_steps = n_steps
        else:
            t_steps = s_steps = n_steps

        t_lib = None
        if wall_rays is not None:
            assert self.t_renderer is not None, \
                'transient renderer does not exist'
            t_lib = self.t_renderer(
                rays=wall_rays, 
                z_vol=feat_vol, 
                n_steps=t_steps, 
                scale=t_scale, 
                sigma_noise=sigma_noise, 
                color_noise=color_noise,
            )

        s_lib = None
        if cam_rays is not None:
            assert self.s_renderer is not None, \
                'steady-state renderer does not exist'
            s_lib = self.s_renderer(
                rays=cam_rays, 
                z_vol=feat_vol, 
                n_steps=s_steps, 
                scale=s_scale, 
                sigma_noise=sigma_noise, 
                color_noise=color_noise,
            )

        return t_lib, s_lib, feat_vol


class EncoderDecoderModel(nn.Module):
    """ Encoder-decoder model for feed-forward inference """
    def __init__(self, encoder, decoder):
        super(EncoderDecoderModel, self).__init__()

        self.encoder = encoder
        self.decoder = decoder

        self.norm = MaxNorm(per_channel=True)

    def forward(self, meas, rot, in_scale=None):
        """
        Args:
            meas (float tensor, (bs, 1/3, t, h, w)): transient measurements.
            rot (float tensor, (v, 3, 3)): rotation matrices.
            in_scale (float): fixed scaling factor for input.

        Returns:
            img (float tensor, (bs, 1/3, h, w)): reconstructed images.
            feat_vol (float tensor, (bs, c, d, h, w)): feature volume.
        """
        if in_scale is None:
            meas = self.norm(meas)
        else:
            meas = meas * in_scale
        
        feat_vol = self.encoder(meas)
        img = self.decoder(feat_vol, rot)
        
        return img, feat_vol