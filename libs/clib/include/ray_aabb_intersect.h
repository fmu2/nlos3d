#pragma once
#include <torch/extension.h>
#include <utility>


std::tuple< at::Tensor, at::Tensor > ray_aabb_intersect(
  at::Tensor ray_o, 
  at::Tensor ray_d, 
  at::Tensor bb_ctr, 
  at::Tensor bb_size);