#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#include "cuda_utils.h"
#include "cutil_math.h"       // required for float3 vector math


__device__ float2 RayAABBIntersection(
  const float3 &ray_o,        // ray origin
  const float3 &ray_d,        // ray direction
  const float3 &bb_ctr,       // AABB center
  const float3 &bb_size) {    // AABB size

  float low = 0, high = 10000;
  float curr_low, curr_high, tmp, inv_dir, r, o, c;

  for (int d = 0; d < 3; ++d) {
    switch (d) {
      case 0:
        inv_dir = __fdividef(1.0f, ray_d.x);
        r = bb_size.x / 2; o = ray_o.x; c = bb_ctr.x;
        break;
      case 1:
        inv_dir = __fdividef(1.0f, ray_d.y);
        r = bb_size.y / 2; o = ray_o.y; c = bb_ctr.y;
        break;
      case 2:
        inv_dir = __fdividef(1.0f, ray_d.z); 
        r = bb_size.z / 2; o = ray_o.z; c = bb_ctr.z;
        break;
    }

    curr_low  = (c - r - o) * inv_dir;
    curr_high = (c + r - o) * inv_dir;

    // swap intersecting distances if necessary
    if (curr_high < curr_low) {
      tmp = curr_low;
      curr_low = curr_high;
      curr_high = tmp;
    }

    // l = max(lx, ly, lz), h = min(hx, hy, hz)
    low = (curr_low > low) ? curr_low : low;
    high = (curr_high < high) ? curr_high : high;

    if (low + 0.001f > high) return make_float2(-1.0f, -1.0f);
  }

  return make_float2(low, high);
}


__global__ void ray_aabb_intersect_kernel(
  int batch_size,                     // batch size
  const float *__restrict__ ray_o,    // ray origin
  const float *__restrict__ ray_d,    // ray direction
  const float *__restrict__ bb_ctr,   // AABB center
  const float *__restrict__ bb_size,  // AABB size
  float *__restrict__ near,           // 1st ray-AABB intersection depth
  float *__restrict__ far) {          // 2nd ray-AABB intersection depth

  int batch_idx = blockIdx.x;
  ray_o += batch_idx * batch_size * 3;
  ray_d += batch_idx * batch_size * 3;
  near += batch_idx * batch_size;
  far += batch_idx * batch_size;

  int thread_idx = threadIdx.x;
  int stride = blockDim.x;

  for (int j = thread_idx; j < batch_size; j += stride) {

    float2 depths = RayAABBIntersection(
      make_float3(ray_o[j * 3], ray_o[j * 3 + 1], ray_o[j * 3 + 2]),
      make_float3(ray_d[j * 3], ray_d[j * 3 + 1], ray_d[j * 3 + 2]),
      make_float3(bb_ctr[0], bb_ctr[1], bb_ctr[2]),
      make_float3(bb_size[0], bb_size[1], bb_size[2]));

    near[j] = depths.x;
    far[j] = depths.y;
  }
}


void ray_aabb_intersect_kernel_wrapper(
  int n_batches,
  int batch_size,
  const float *ray_o,
  const float *ray_d,
  const float *bb_ctr,
  const float *bb_size,
  float *near,
  float *far) {

  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  ray_aabb_intersect_kernel<<<n_batches, opt_n_threads(batch_size), 0, stream>>>(
    batch_size, ray_o, ray_d, bb_ctr, bb_size, near, far);

  CUDA_CHECK_ERRORS();
}