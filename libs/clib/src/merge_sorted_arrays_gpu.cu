#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#include "cuda_utils.h"


__global__ void merge_sorted_arrays_kernel(
  int batch_size,                    // batch size
  int n1,                            // length of 1st array
  int n2,                            // length of 2nd array
  const float *__restrict__ src1,    // 1st array
  const float *__restrict__ src2,    // 2nd array
  int *__restrict__ labels,          // 0 if from 1st array, 1 otherwise
  float *__restrict__ out) {         // merged array

  int batch_idx = blockIdx.x;
  int n = n1 + n2;

  src1 += batch_idx * batch_size * n1;
  src2 += batch_idx * batch_size * n2;
  labels += batch_idx * batch_size * n;
  out += batch_idx * batch_size * n;

  int thread_idx = threadIdx.x;
  int stride = blockDim.x;

  for (int j = thread_idx; j < batch_size; j += stride) {
    int p1 = 0;
    int p2 = 0;
    int k = 0;

    while (p1 < n1 && p2 < n2) {
      if (src1[j * n1 + p1] < src2[j * n2 + p2]) {
        labels[j * n + k] = 0;
        out[j * n + k] = src1[j * n1 + p1];
        ++p1;
      }
      else {
        labels[j * n + k] = 1;
        out[j * n + k] = src2[j * n2 + p2];
        ++p2;
      }
      ++k;
    }

    if (p1 < n1) {
      while (p1 < n1) {
        labels[j * n + k] = 0;
        out[j * n + k] = src1[j * n1 + p1];
        ++p1; ++k;
      }
    }
    else {
      while (p2 < n2) {
        labels[j * n + k] = 1;
        out[j * n + k] = src2[j * n2 + p2];
        ++p2; ++k;
      }
    }
  }
}


void merge_sorted_arrays_kernel_wrapper(
  int n_batches,
  int batch_size,
  int n1,
  int n2,
  const float *src1,
  const float *src2,
  int *labels,
  float *out) {

  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  merge_sorted_arrays_kernel<<<n_batches, opt_n_threads(batch_size), 0, stream>>>(
    batch_size, n1, n2, src1, src2, labels, out);

  CUDA_CHECK_ERRORS();
}