#include "ray_aabb_intersect.h"
#include "merge_sorted_arrays.h"


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("ray_aabb_intersect", &ray_aabb_intersect);
  m.def("merge_sorted_arrays", &merge_sorted_arrays);
}