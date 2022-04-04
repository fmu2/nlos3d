import math

import cv2
import numpy as np


trajectories = {}

def register_trajectory(name):
    def decorator(fn):
        if name in trajectories:
            raise ValueError(
                "trajectory already exists: {:s}".format(name)
            )
        trajectories[name] = fn
        return fn
    return decorator


def make_trajectory(config):
    try:
        name = config["type"]
        return trajectories[name](**config[name])
    except:
        raise ValueError(
            "trajectory does not exist: {:s}".format(config["type"])
        )


@register_trajectory("spiral")
def spiral(axis="z", ctr=[0, 0, 0], r0=3., r1=1.5, h0=0.5, h1=0.25):
    """
    Spiral trajectory.

    Args:
        axis (str): rotation axis.
        ctr (float 3-tuple): center of reference frame.
        r0 (float): radius to begin with.
        r1 (float): radius to end with.
        h0 (float): height to begin with.
        h1 (float): height to end with.

    Returns:
        A function that takes as input a value from [0, 1] and outputs the 
        xyz-coordinates of a 3D point.
    """
    assert isinstance(ctr, (list, tuple)) and len(ctr) == 3
    r = lambda t: r1 + (r0 - r1) * t
    h = lambda t: h1 + (h0 - h1) * t
    
    if axis == "x":
        return lambda t: [h(t) + ctr[0], 
                          r(t) * math.cos(2 * math.pi * t) + ctr[1], 
                          r(t) * math.sin(2 * math.pi * t) + ctr[2]]
    elif axis == "y":
        return lambda t: [r(t) * math.cos(2 * math.pi * t) + ctr[0], 
                          h(t) + ctr[1], 
                          r(t) * math.sin(2 * math.pi * t) + ctr[2]]
    elif axis == "z":
        return lambda t: [r(t) * math.cos(2 * math.pi * t) + ctr[0], 
                          r(t) * math.sin(2 * math.pi * t) + ctr[1], 
                          h(t) + ctr[2]]
    else:
        raise ValueError("invalid rotation axis: {:s}".format(axis))


@register_trajectory("circle")
def circle(axis="z", ctr=[0, 0, 0], r=3, h=0.5):
    """
    Circular trajectory.

    Args:
        axis (str): rotation axis.
        ctr (float 3-tuple): center of reference frame.
        r (float): radius of the circle.
        h (float): height of the circle.

    Returns:
        A function that takes as input a value from [0, 1] and outputs the 
        xyz-coordinates of a 3D point.
    """
    return spiral(axis, ctr, r, r, h, h)


@register_trajectory("zoom")
def zoom(axis="z", ctr=[0, 0, 0], r0=3., r1=1.5, d0=0.5, d1=0.25, alpha=0.):
    """
    A trajectory that zooms in or out on the scene.

    Args:
        axis (str): axis along which to zoom in or out.
        ctr (float 3-tuple): center of reference frame.
        r0 (float): radius to begin with.
        r1 (float): radius to end with.
        d0 (float): depth to begin with.
        d1 (float): depth to end with.
        alpha (float): a scalar that controls rotation speed.

    Returns:
        A function that takes as input a value from [0, 1] and outputs the 
        xyz-coordinates of a 3D point.
    """
    assert isinstance(ctr, (list, tuple)) and len(ctr) == 3
    r = lambda t: r1 + (r0 - r1) * t
    h = lambda t: h1 + (h0 - h1) * t

    if axis == "x":
        return lambda t: [h(t) + ctr[0], 
                          r(t) * math.cos(2 * math.pi * alpha) + ctr[1], 
                          r(t) * math.sin(2 * math.pi * alpha) + ctr[2]]
    elif axis == "y":
        return lambda t: [r(t) * math.cos(2 * math.pi * alpha) + ctr[0], 
                          h(t) + ctr[1], 
                          r(t) * math.sin(2 * math.pi * alpha) + ctr[2]]
    elif axis == "z":
        return lambda t: [r(t) * math.cos(2 * math.pi * alpha) + ctr[0], 
                          r(t) * math.sin(2 * math.pi * alpha) + ctr[1], 
                          h(t) + ctr[2]]
    else:
        raise ValueError("invalid zooming axis: {:s}".format(axis))


def make_camera(config):
    name = config["type"]
    if name == "perspective":
        return PerspectiveCamera(**config[name])
    elif name == "orthographic":
        return OrthographicCamera(**config[name])
    else:
        raise ValueError("invalid camera type: {:s}".format(name))


def rodrigues(ctr, rot_vec, world=True):
    """
    Construct world-to-camera transformation using Rodrigues' algorithm.

    Args:
        ctr (float 3-tuple): camera center.
            (NOTE: this is the image center for an orthographic camera.)
        rot_vec (float 3-tuple): rotation vector.
        world (bool): if True, the given camera center is in world frame.
            Otherwise, it is in the frame after rotation.

    Returns:
        Rt (float array, [4, 4]): world-to-camera transformation.
    """
    ctr = np.array(ctr, dtype=float)
    rot_vec = np.array(rot_vec, dtype=float)

    Rt = np.eye(4)
    Rt[:3, :3] = cv2.Rodrigues(rot_vec)[0]
    if world:
        Rt[:3, 3] = np.matmul(Rt[:3, :3], -ctr)
    else:
        Rt[:3, 3] = -ctr
    return Rt


def look_at(ctr, at, up=None, world=True):
    """ 
    Construct world-to-camera transformation using the look-at method.

    Args:
        ctr (float 3-tuple): camera center in the world frame.
            (NOTE: this is the image center for an orthographic camera.)
        at (float 3-tuple): look-at location in the world frame.
        up (float 3-tuple): up direction in the world frame.
        world (bool): if True, the given camera center is in world frame.
            Otherwise, it is in the frame after rotation.

    Returns:
        Rt (float array, (4, 4)): world-to-camera transformation.
    """
    if up is None:
        up = [0, 1, 0]
    ctr = np.array(ctr, dtype=float)
    at = np.array(at, dtype=float)
    up = np.array(up, dtype=float)

    z = ctr - at                    # forward (into the camera)
    x = np.cross(up, z)             # right
    y = np.cross(z, x)              # up 
    
    z /= np.linalg.norm(z)
    x /= np.linalg.norm(x)
    y /= np.linalg.norm(y)
    
    Rt = np.eye(4)
    Rt[:3, :3] = np.stack([x, y, z], -1)
    if world:
        Rt[:3, 3] = np.matmul(Rt[:3, :3], -ctr)
    else:
        Rt[:3, 3] = -ctr
    return Rt


def get_all_views(r=2, n_views=26, ratio=0.25):
    """
    Obtain camera views for multi-view supervision.

    NOTE: The camera principal axes are drawn using the Fibonacci spiral 
    algorithm. The implementation follows Chen et al., SIGGRAPH Asia 20.

    Args:
        r (float): radius of circular camera trajectory.
        n_views (int): number of views including the canonical view.
        ratio (float): ratio that controls the span of rotations.

    Returns:
        rot_vecs (float array, (v, 3, 3)): world-to-camera transformations.
    """
    z_axis = np.array([0., 0., 1.])
    n_max = (n_views - 1) / ratio
    n_min = n_max - (n_views - 1)
    
    idx = np.arange(n_min + 1, n_max + 1)
    sinp = idx / (n_max + 1)
    cosp = np.sqrt(1 - sinp ** 2)

    igr = (5 ** 0.5 - 1) / 2                            # inverse golden ratio
    ga = 2 * np.pi * igr                                # golden angle
    t = ga * idx                                        # longitude

    x = cosp * np.cos(t)
    y = cosp * np.sin(t)
    z = sinp

    p_axis = np.stack([x, y, z], -1)                    # camera principal axis
    rv = np.cross(z_axis, p_axis)                       # rotation vector
    rv /= np.linalg.norm(rv, axis=-1, keepdims=True)    # normalized rotation vector
    ra = np.arccos(z)                                   # rotation angle
    
    rot_vecs = ra[:, None] * rv
    rot_vecs = np.concatenate([np.zeros((1, 3)), rot_vecs], 0)

    Rt = []
    for i in range(n_views):
        Rt.append(rodrigues([0, 0, r], rot_vecs[i], world=False))
    Rt = np.stack(Rt, 0)                                # (v, 3, 3)
    return Rt

Rt_all = get_all_views()


def sample_views(n_views=1, include_orthogonal=True):
    """
    Sample camera views.

    Args:
        n_views (int): number of camera views.
        include_orthogonal (bool): if True, always include the orthogonal view.

    Returns:
        idx (int list): sampled view indices.
        Rt (float array, (v, 3)): sampled camera views.
    """
    if include_orthogonal:
        idx = np.random.choice(
            np.arange(1, len(Rt_all)), n_views - 1, replace=False
        )
        idx = [0] + list(idx)
    else:
        idx = np.random.choice(len(Rt_all), n_views, replace=False)
    Rt = Rt_all[idx]
    return idx, Rt


class PerspectiveCamera:
    """
    Perspective pinhole camera
    (NOTE: this implementation assumes RIGHT-handed system)
    """
    def __init__(
        self, 
        size=0.04, 
        f=0.02, 
        res=256, 
        block_size=4,
        foreshortening=False,
    ):
        """
        Args:
            size (float): physical size of image sensor (unit: m).
            f (float): effective focal length (unit: m).
            res (int): image resolution (unit: px).
            block_size (int): pixel block size (unit: px).
            foreshortening (bool): if True, account for cosine attenuation.
        """
        # camera intrinsics
        self.size = size
        self.f = f
        self.res = res

        # pixel blocks for sampling
        self.block_size = block_size
        assert res % block_size == 0, \
            "sensor grid resolution must be divisible by block size"
        self.n_blocks = (res // block_size) ** 2  # number of pixel blocks
        self.p = block_size ** 2                  # number of pixels per block

        self.foreshortening = foreshortening

        # ij-indices of top-left corners of pixel blocks
        tics = np.arange(0, res, block_size)
        j, i = np.meshgrid(tics, tics)
        self.bk_corners = np.stack([i.flatten(), j.flatten()], -1) # [B, 2]

        # pixel uv-coordinates in image frame
        ## NOTE: origin of image frame is at the bottom-left corner.
        tics = np.linspace(0, size, res + 1)
        tics = (tics[:-1] + tics[1:]) / 2     # mid-point rule
        u, v = np.meshgrid(tics, tics[::-1])

        # pixel xyz-coordinates in camera frame
        x = u - size / 2
        y = v - size / 2
        z = -f * np.ones_like(x)
        self.px_centers = np.stack([x, y, z], axis=-1)            # (h, w, 3)

    def _prepare_rays(self, Rt, idx=None, invert_z=False):
        # xyz-coordinates of sampled pixels in camera frame
        if idx is None:
            o = self.px_centers.reshape(-1, 3)
        else:
            o = self.px_centers[idx[:, 0], idx[:, 1]]

        # xyz-coordinates of sampled pixels in world frame
        o = np.concatenate([o, np.ones_like(o[:, :1])], -1)       # (n, 4)
        o = np.matmul(o, np.linalg.inv(Rt).T)                     # (n, 4)
        o = o[:, :3] / o[:, 3:]                                   # (n, 3)

        # ray directions in world frame
        r = o + np.matmul(Rt[:3, :3].T, Rt[:3, 3])
        r /= np.linalg.norm(r, axis=-1, keepdims=True)            # (n, 3)

        # ray weights
        if self.foreshortening:
            normal = np.matmul(Rt[:3, :3].T, np.array([0., 0., -1.]))
            w = np.dot(r, self.normal)[:, None]  # cosine term    # (n, 1)
            w = np.clip(w, 0, 1)
        else:
            w = np.ones_like(r[:, :1])                            # (n, 1)

        if invert_z:
            o[..., -1] *= -1
            r[..., -1] *= -1

        rays = np.concatenate([o, r, w], -1)                      # (n, 7)
        return rays

    def sample_rays(self, Rt, batch_size=1, invert_z=False):
        """
        Return rays originating from sampled pixels.
        (NOTE: the rays are in the world-frame.)

        Args:
            Rt (float array, (4, 4)): world-to-camera transformation.
            batch_size (int): number of batches.
            invert_z (bool): if True, switch to LEFT-handed system.

        Returns:
            idx (int array, (k, r, 2)): sampled pixel indices.
            rays (float array, (k, r, 7)): ray (origin, direction, weight) bundles.
        """
        k, r = batch_size, self.n_blocks

        # sample pixel indices
        s = np.random.choice(self.p, (k, r))
        d = np.stack([s // self.block_size, s % self.block_size], -1)
        idx = self.bk_corners + d                                 # (k, r, 2)
        
        rays = self._prepare_rays(Rt, idx.reshape(k * r, 2), invert_z)
        rays = rays.reshape(k, r, 7)
        return idx, rays

    def get_all_rays(self, Rt, invert_z=False):
        """
        Return all rays.

        Args:
            Rt (float array, (4, 4)): world-to-camera transformation.
            invert_z (bool): if True, switch to LEFT-handed system.

        Returns:
            rays (float array, (h, w, 7)): ray (origin, direction, weight) bundle.
        """
        rays = self._prepare_rays(Rt, None, invert_z)
        rays = rays.reshape(self.res, self.res, 7)
        return rays


class OrthographicCamera:
    """
    Orthographic camera
    (NOTE: this implementation assumes RIGHT-handed system)
    """
    def __init__(
        self, 
        size=2, 
        res=256, 
        block_size=4,
    ):
        """
        Args:
            size (float): physical size of image sensor (unit: m).
            res (int): image resolution (unit: px).
            block_size (int): pixel block size (unit: px).
        """
        # intrinsics
        self.size = size
        self.res = res

        # pixel blocks for sampling
        self.block_size = block_size
        assert res % block_size == 0, \
            "sensor grid resolution must be divisible by block size"
        self.n_blocks = (res // block_size) ** 2  # number of pixel blocks
        self.p = block_size ** 2                  # number of pixels per block

        # ij-indices of top-left corners of pixel blocks
        tics = np.arange(0, res, block_size)
        j, i = np.meshgrid(tics, tics)
        self.bk_corners = np.stack([i.flatten(), j.flatten()], -1) # (b, 2)

        # pixel uv-coordinates in image frame
        ## NOTE: origin of image frame is at the bottom-left corner.
        tics = np.linspace(0, size, res + 1)
        tics = (tics[:-1] + tics[1:]) / 2     # mid-point rule
        u, v = np.meshgrid(tics, tics[::-1])

        # xy-coordinates of pixels in camera frame
        x = u - size / 2
        y = v - size / 2
        z = np.zeros_like(x)
        self.px_centers = np.stack([x, y, z], -1)                 # (h, w, 3)

    def _prepare_rays(self, Rt, idx=None, invert_z=False):
        # xyz-coordinates of sampled pixels in camera frame
        if idx is None:
            o = self.px_centers.reshape(-1, 3)
        else:
            o = self.px_centers[idx[:, 0], idx[:, 1]]

        # xyz-coordinates of sampled pixels in world frame
        o = np.concatenate([o, np.ones_like(o[:, :1])], -1)       # (n, 4)
        o = np.matmul(o, np.linalg.inv(Rt).T)                     # (n, 4)
        o = o[:, :3] / o[:, 3:]                                   # (n, 3)

        # ray directions in world frame
        r = np.matmul(Rt[:3, :3].T, np.array([0., 0., -1.]))
        r = r[None].repeat(len(o), 0)                             # (n, 3)

        # ray weights
        w = np.ones_like(r[:, :1])                                # (n, 1)

        if invert_z:
            o[..., -1] *= -1
            r[..., -1] *= -1

        rays = np.concatenate([o, r, w], -1)                      # (n, 7)
        return rays

    def sample_rays(self, Rt, batch_size=1, invert_z=False):
        """
        Return rays originating from sampled pixels.
        (NOTE: the rays are in the world-frame.)

        Args:
            Rt (float array, (4, 4)): world-to-camera transformation.
            batch_size (int): number of batches.
            invert_z (bool): if True, switch to LEFT-handed system.

        Returns:
            idx (int array, (k, r, 2)): sampled pixel indices.
            rays (float array, (k, r, 7)): ray (origin, direction, weight) bundles.
        """
        k, r = batch_size, self.n_blocks

        # sample pixel indices
        s = np.random.choice(self.p, (k, r))
        d = np.stack([s // self.block_size, s % self.block_size], -1)
        idx = self.bk_corners + d                                 # (k, r, 2)
        
        rays = self._prepare_rays(Rt, idx.reshape(k * r, 2), invert_z)
        rays = rays.reshape(k, r, 7)
        return idx, rays

    def get_all_rays(self, Rt, invert_z=False):
        """
        Return all rays.

        Args:
            Rt (float array, (4, 4)): world-to-camera transformation.
            invert_z (bool): if True, switch to LEFT-handed system.

        Returns:
            rays (float array, (h, w, 7)): ray (origin, direction, weight) bundle.
        """
        rays = self._prepare_rays(Rt, None, invert_z)
        rays = rays.reshape(self.res, self.res, 7)
        return rays


def make_wall(config):
    return Wall(**config)


class Wall:
    """
    Relay wall perpendicular to z-axis of world frame and centering at (0, 0, 1)
    (NOTE: this implementation assumes RIGHT-handed system)
    """
    def __init__(
        self, 
        size=2, 
        res=64, 
        block_size=32, 
        uv_size=64,
        foreshortening=False, 
        depth=1,
    ):
        """
        Args:
            size (float): physical size of the relay wall (unit: m).
            res (int): spatial resolution of virtual sensor grid (unit: px).
            block_size (int): pixel block size (unit: px).
            uv_size (int): size of ray-sampling grid (unit: px).
            foreshortening (bool): if True, account for cosine attenuation.
            depth (float): distance from wall to the near plane of reconstruction.

        (b: number of blocks, 
         d: sensor grid resolution, 
         u: UV-grid resolution,
        )
        """
        self.size = size
        self.res = res

        # pixel blocks for sampling
        self.block_size = block_size
        assert res % block_size == 0, \
            "sensor grid resolution must be divisible by block size"
        self.n_blocks = (res // block_size) ** 2  # number of pixel blocks
        self.p = block_size ** 2                  # number of pixels per block

        self.uv_size = uv_size
        self.u = uv_size ** 2                     # number of cells in the UV-grid

        self.foreshortening = foreshortening

        assert depth > 0
        self.near = 1 - depth                     # z-coordinate of near plane

        # pixel xyz-coordinates in world frame
        tics = np.linspace(-size / 2, size / 2, res + 1)
        tics = (tics[:-1] + tics[1:]) / 2
        x, y = np.meshgrid(tics, tics[::-1])
        z = np.ones_like(x)
        self.px_centers = np.stack([x, y, z], -1)                  # (d, d, 3)

        # ij-indices of top-left corners of pixel blocks
        tics = np.arange(0, res, block_size)
        j, i = np.meshgrid(tics, tics)
        self.bk_corners = np.stack([i.flatten(), j.flatten()], -1) # (b, 2)

        # uv-coordinates of bottom-left corners of UV-grid cells
        tics = np.linspace(0, 1, uv_size + 1)[:-1]
        x, y = np.meshgrid(tics, tics[::-1])
        self.uv_corners = np.stack([x.flatten(), y.flatten()], -1) # (u*u, 2)

    def sample_rays(self, batch_size=1, invert_z=False):
        """
        Return rays originating from the sampled pixels. The rays from each pixel 
        are drawn uniformly at random from the projection of the near plane of 
        reconstruction volume onto the unit sphere centered at wall center.

        Args:
            batch_size (int): number of batches.
            invert_z (bool): if True, switch to LEFT-handed system.

        Returns:
            idx (int array, (k, v, 2)): pixel indices.
            rays (float array, (k, v, n, 7)): ray (origin, direction and weight) bundles.

        (k: batch size, 
         v: number of views, 
         n: number of rays per view,
        )
        """
        K, V = batch_size, self.n_blocks

        # sample pixel indices
        s = np.random.choice(self.p, (K, V))
        d = np.stack([s // self.block_size, s % self.block_size], -1)
        idx = self.bk_corners + d                                   # (k, v, 2)

        # xyz-coordinates of sampled pixels (i.e., ray origins)
        idx = idx.reshape(K * V, 2)                                 # (k*v, 2)
        o = self.px_centers[idx[:, 0], idx[:, 1]]                   # (k*v, 3)

        # sample ray directions (Urena et al., EuroGraphics 13)
        bl = np.array(   # bottom-left corner of near plane
            [-self.size / 2, -self.size / 2, self.near], 
            dtype=float,
        )
        p = bl - o                                                  # (k*v, 3)
        x0, y0, z0 = p[:, 0:1], p[:, 1:2], p[:, 2:3]                # (k*v, 1)
        x1, y1 = x0 + self.size, y0 + self.size                     # (k*v, 1)
        v00 = np.concatenate([x0, y0, z0], -1)                      # (k*v, 3)
        v01 = np.concatenate([x0, y1, z0], -1)
        v10 = np.concatenate([x1, y0, z0], -1)
        v11 = np.concatenate([x1, y1, z0], -1)

        n0 = np.cross(v00, v10)                                     # (k*v, 3)
        n1 = np.cross(v10, v11)                                     # (k*v, 3)
        n2 = np.cross(v11, v01)                                     # (k*v, 3)
        n3 = np.cross(v01, v00)                                     # (k*v, 3)
        n0 /= np.linalg.norm(n0, axis=-1, keepdims=True)
        n1 /= np.linalg.norm(n1, axis=-1, keepdims=True)
        n2 /= np.linalg.norm(n2, axis=-1, keepdims=True)
        n3 /= np.linalg.norm(n3, axis=-1, keepdims=True)

        g0 = np.arccos(-np.einsum("ij,ij->i", n0, n1))[:, None]     # (k*v, 1)
        g1 = np.arccos(-np.einsum("ij,ij->i", n1, n2))[:, None]     # (k*v, 1)
        g2 = np.arccos(-np.einsum("ij,ij->i", n2, n3))[:, None]     # (k*v, 1)
        g3 = np.arccos(-np.einsum("ij,ij->i", n3, n0))[:, None]     # (k*v, 1)

        b0, b1 = n0[:, -1:], n2[:, -1:]
        k = 2 * np.pi - g2 - g3                                     # (k*v, 1)
        S = g0 + g1 - k    # solid angle                            # (k*v, 1)

        uv = np.random.rand(K, V, self.u, 2)                        # (k, v, n, 2)
        uv = uv / self.uv_size + self.uv_corners
        uv = uv.reshape(K * V, self.u, 2)                           # (k*v, n, 2)
        u, v = uv[..., 0], uv[..., 1]                               # (k*v, n)

        au = u * S + k
        fu = (np.cos(au) * b0 - b1) / np.sin(au)
        cu = ((fu > 0) * 2 - 1) / np.sqrt(fu ** 2 + b0 ** 2)        # (k*v, n)
        cu = np.clip(cu, -1, 1)

        xu = -cu * z0 / np.sqrt(1 - cu ** 2)                        # (k*v, n)
        xu = np.clip(xu, x0, x1)

        d = np.sqrt(xu ** 2 + z0 ** 2)                              # (k*v, n)
        h0 = y0 / np.sqrt(d ** 2 + y0 ** 2)                         # (k*v, n)
        h1 = y1 / np.sqrt(d ** 2 + y1 ** 2)                         # (k*v, n)
        hv = h0 + v * (h1 - h0)                                     # (k*v, n)
        yv = (hv * d) / np.sqrt(1 - hv ** 2 + 1e-8)                 # (k*v, n)
        yv = np.clip(yv, y0, y1)

        r = np.stack([xu, yv, z0.repeat(self.u, -1)], -1)           # (k*v, n, 3)
        r /= np.linalg.norm(r, axis=-1, keepdims=True)

        # ray weights
        if self.foreshortening:
            w = np.dot(r, np.array([0., 0., -1.])) # cosine term    # (k*v, n)
            w = np.clip(w, 0, 1)
        else:
            w = np.ones_like(u)
        w = w * S / self.u             # importance weights
        w = w[..., None]                                            # (k*v, n, 1)

        # invert z-axis so that the world frame follows LEFT-hand rule
        if invert_z:
            o[..., -1] *= -1
            r[..., -1] *= -1

        rays = np.concatenate(
            [o[:, None].repeat(self.u, 1), r, w], -1)               # (k*v, n, 7)
        
        idx = idx.reshape(K, V, 2)
        rays = rays.reshape(K, V, self.u, 7)
        return idx, rays