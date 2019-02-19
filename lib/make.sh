#!/usr/bin/env bash


CUDA_PATH=/usr/local/cuda/

python3 setup.py build_ext --inplace
rm -rf build
