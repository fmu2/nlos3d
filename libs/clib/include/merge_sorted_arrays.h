#pragma once
#include <torch/extension.h>
#include <utility>


std::tuple< at::Tensor, at::Tensor > merge_sorted_arrays(
  at::Tensor src1, 
  at::Tensor src2);