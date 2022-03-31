import math

import torch
from torch.autograd import Function

try:
  from . import _ext
except ImportError:
  raise ImportError(
    "Failed to import _ext module.\n"
    "Run 'python setup.py build_ext --inplace' first."
  )

MAX_DEPTH = 1.e4


class RayAABBIntersect(Function):
  """ Performs ray-AABB intersection tests. """
  @staticmethod
  def forward(ctx, ray_o, ray_d, bb_ctr, bb_size):
    """
    Args:
      ray_o (float tensor, [R, 3]): ray origins.
      ray_d (float tensor, [R, 3]): ray directions.
      bb_ctr (float tensor, [3,]): AABB center.
      bb_size (float tensor, [3,]): AABB size (unit: m).

    Returns:
      near (float tensor, [R, 1]): 1st ray-AABB intersection depth.
      far (float tensor, [R, 1]): 2nd ray-AABB intersection depth.
    """
    ray_o, ray_d = ray_o.contiguous(), ray_d.contiguous()
    assert ray_o.size() == ray_d.size()
    N = ray_o.size(0)

    # divide into batches for GPU parallelization
    batch_size = min(N, 4096)
    n_batches = int(math.ceil(N / batch_size))
    H = n_batches * batch_size

    if H > N:   # pad all batches to same size
      ray_o = torch.cat([ray_o, ray_o[:H - N]])
      ray_d = torch.cat([ray_d, ray_d[:H - N]])
    
    ray_o = ray_o.reshape(n_batches, batch_size, 3)
    ray_d = ray_d.reshape(n_batches, batch_size, 3)

    near, far = _ext.ray_aabb_intersect(
      ray_o.float(), ray_d.float(), bb_ctr.float(), bb_size.float())
    near, far = near.type_as(ray_o), far.type_as(ray_o)

    # recover shape and remove padding
    near = near.reshape(H, -1)[:N]
    far = far.reshape(H, -1)[:N]

    ctx.mark_non_differentiable(near)
    ctx.mark_non_differentiable(far)

    return near, far

  @staticmethod
  def backward(ctx, a, b):
    return None, None, None, None


ray_aabb_intersect = RayAABBIntersect.apply


class MergeSortedArrays(Function):
  """ Merges two sorted arrays of depth values. """
  @staticmethod
  def forward(ctx, src1, src2):
    """
    Args:
      src1 (float tensor, [R, N1]): 1st array.
      src2 (float tensor, [R, N2]): 2nd array.

    Returns:
      labels (int tensor, [R, N1 + N2]): ID of array an element belongs to.
        (0 for 1st array, 1 for 2nd array)
      out (float tensor, [R, N1 + N2]): merged array.
    """
    src1, src2 = src1.contiguous(), src2.contiguous()
    assert src1.size(0) == src2.size(0)
    N = src1.size(0)

    # divide into batches for GPU parallelization
    batch_size = min(N, 4096)
    n_batches = int(math.ceil(N / batch_size))
    H = n_batches * batch_size

    if H > N:   # pad all batches to same size
      src1 = torch.cat([src1, src1[:H - N]])
      src2 = torch.cat([src2, src2[:H - N]])
    
    src1 = src1.reshape(n_batches, batch_size, -1)
    src2 = src2.reshape(n_batches, batch_size, -1)

    labels, out = _ext.merge_sorted_arrays(src1.float(), src2.float())
    out = out.type_as(src1)

    # recover shape and remove padding
    labels = labels.reshape(H, -1)[:N]
    out = out.reshape(H, -1)[:N]

    ctx.mark_non_differentiable(labels)
    ctx.mark_non_differentiable(out)

    return labels, out

  @staticmethod
  def backward(ctx, a, b):
    return None, None


merge_sorted_arrays = MergeSortedArrays.apply