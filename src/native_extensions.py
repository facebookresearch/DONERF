# Copyright (c) Facebook, Inc. All Rights Reserved
from setuptools import setup
from torch.utils import cpp_extension

setup(name='disc_depth_multiclass',
      ext_modules=[cpp_extension.CUDAExtension('disc_depth_multiclass', [
            './src/native/disc_depth_multiclass.cpp',
            './src/native/disc_depth_multiclass_cuda.cu'
            ])],
      version="0.0.2",
      cmdclass={'build_ext': cpp_extension.BuildExtension})
