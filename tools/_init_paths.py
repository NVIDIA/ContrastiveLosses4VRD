# Based on Detectron.pytorch/tools/_init_paths.py by Roy Tseng
# modified for this project by Ji Zhang

"""Add {PROJECT_ROOT}/lib. to PYTHONPATH

Usage:
import this module before import any modules under lib/
e.g 
    import _init_paths
    from core.config import cfg
""" 

import os.path as osp
import sys


def add_path(path):
    if path not in sys.path:
        sys.path.insert(0, path)

this_dir = osp.abspath(osp.dirname(osp.dirname(__file__)))

# add Detectron.PyTorch/lib
detectron_path = osp.join(this_dir, 'Detectron_pytorch', 'lib')
add_path(detectron_path)

# Add lib to PYTHONPATH
lib_path = osp.join(this_dir, 'lib')
add_path(lib_path)

