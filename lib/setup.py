# Based on:
# Detectron.pytorch/lib/setup.py
# and modified for this project
# Original source license text:
# --------------------------------------------------------
# Fast R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# --------------------------------------------------------

from __future__ import print_function

from Cython.Build import cythonize
from Cython.Distutils import build_ext
from setuptools import Extension
from setuptools import setup

import numpy as np


# Obtain the numpy include directory.  This logic works across numpy versions.
try:
    numpy_include = np.get_include()
except AttributeError:
    numpy_include = np.get_numpy_include()


ext_modules = [
    Extension(
        name='utils_rel.cython_bbox_rel',
        sources=['utils_rel/cython_bbox_rel.pyx'],
        extra_compile_args=['-Wno-cpp'],
        include_dirs=[numpy_include]
    )
]

setup(
    name='mask_rcnn_rel',
    ext_modules=cythonize(ext_modules)
)

