#include "merge_sorted_arrays.h"
#include "utils.h"
#include <utility>


void merge_sorted_arrays_kernel_wrapper(
  int n_batches,
  int batch_size,
  int n1,
  int n2,
  const float *src1,
  const float *src2,
  int *labels,
  float *out);


std::tuple< at::Tensor, at::Tensor > merge_sorted_arrays(
  at::Tensor src1, 
  at::Tensor src2) {

  CHECK_CONTIGUOUS(src1);
  CHECK_CONTIGUOUS(src2);

  CHECK_IS_FLOAT(src1);
  CHECK_IS_FLOAT(src2);

  CHECK_CUDA(src1);
  CHECK_CUDA(src2);

  at::Tensor labels = torch::zeros(
    {src1.size(0), src1.size(1), src1.size(2) + src2.size(2)},
    at::device(src1.device()).dtype(at::ScalarType::Int));
  at::Tensor out = torch::zeros(
    {src1.size(0), src1.size(1), src1.size(2) + src2.size(2)},
    at::device(src1.device()).dtype(at::ScalarType::Float));

  merge_sorted_arrays_kernel_wrapper(
    src1.size(0),
    src1.size(1),
    src1.size(2),
    src2.size(2),
    src1.data_ptr <float>(),
    src2.data_ptr <float>(),
    labels.data_ptr <int>(),
    out.data_ptr <float>());

  return std::make_tuple(labels, out);
}