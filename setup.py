import os
import glob
from setuptools import setup

from torch.utils.cpp_extension import BuildExtension, CUDAExtension


_ext_src_root = os.path.abspath("libs/clib")
_ext_sources = glob.glob("{}/src/*.cpp".format(_ext_src_root)) + glob.glob(
  "{}/src/*.cu".format(_ext_src_root)
)
_ext_headers = glob.glob("{}/include/*".format(_ext_src_root))

setup(
  name='nlos',
  ext_modules=[
    CUDAExtension(
      name='libs.clib._ext',
      sources=_ext_sources,
      extra_compile_args={
        "cxx": ["-O2", "-I{}".format("{}/include".format(_ext_src_root))],
        "nvcc": ["-O2", "-I{}".format("{}/include".format(_ext_src_root))],
      },
    )
  ],
  cmdclass={
    'build_ext': BuildExtension
  }
)
