#include "ray_aabb_intersect.h"
#include "utils.h"
#include <utility>


void ray_aabb_intersect_kernel_wrapper(
  int n_batches,
  int batch_size,
  const float *ray_o,
  const float *ray_d,
  const float *bb_ctr,
  const float *bb_size,
  float *near,
  float *far);


std::tuple< at::Tensor, at::Tensor > ray_aabb_intersect(
  at::Tensor ray_o, 
  at::Tensor ray_d, 
  at::Tensor bb_ctr, 
  at::Tensor bb_size) {

  CHECK_CONTIGUOUS(ray_o);
  CHECK_CONTIGUOUS(ray_d);
  CHECK_CONTIGUOUS(bb_ctr);
  CHECK_CONTIGUOUS(bb_size);

  CHECK_IS_FLOAT(ray_o);
  CHECK_IS_FLOAT(ray_d);
  CHECK_IS_FLOAT(bb_ctr);
  CHECK_IS_FLOAT(bb_size);

  CHECK_CUDA(ray_o);
  CHECK_CUDA(ray_d);
  CHECK_CUDA(bb_ctr);
  CHECK_CUDA(bb_size);

  at::Tensor near = torch::zeros(
    {ray_o.size(0), ray_o.size(1)},
    at::device(ray_o.device()).dtype(at::ScalarType::Float));
  at::Tensor far = torch::zeros(
    {ray_o.size(0), ray_o.size(1)},
    at::device(ray_o.device()).dtype(at::ScalarType::Float));

  ray_aabb_intersect_kernel_wrapper(
    ray_o.size(0),
    ray_o.size(1),
    ray_o.data_ptr <float>(),
    ray_d.data_ptr <float>(),
    bb_ctr.data_ptr <float>(),
    bb_size.data_ptr <float>(),
    near.data_ptr <float>(),
    far.data_ptr <float>());

  return std::make_tuple(near, far);
}