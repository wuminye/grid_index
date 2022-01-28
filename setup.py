'''
@author:  Minye Wu
@contact: wuminye.x@gmail.com
'''
import os
import torch

os.environ['CUDA_VISIBLE_DEVICES']='0'
print(os.environ['CUDA_VISIBLE_DEVICES'])
print(torch.cuda.is_available())
print(torch.cuda.get_device_name(0))

from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

extra_compile_args = {"cxx": [
     "-Wdeprecated-declarations"
    ]}

setup(
    name='grid_index',
    version="0.2",
    ext_modules=[
        CUDAExtension('grid_index', [
            'src/grid_index.cpp',
            'src/grid_index_gpu.cu',
           
        ])
    ],
    cmdclass={
        'build_ext': BuildExtension.with_options(use_ninja=False)
    })
