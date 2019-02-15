# Adapted by Ji Zhang, 2019
#
# Based on test_net.py Written by Roy Tseng

"""Perform inference on one or more datasets."""

import argparse
import cv2
import os
import pprint
import sys
import time
from six.moves import cPickle as pickle

import torch

import _init_paths  # pylint: disable=unused-import
from core.config import cfg, merge_cfg_from_file, merge_cfg_from_list, assert_and_infer_cfg
from core.test_engine_rel import run_inference
import utils_rel.logging

from datasets_rel import task_evaluation_sg_rel_and_att

# OpenCL may be enabled by default in OpenCV3; disable it because it's not
# thread safe and causes unwanted GPU memory allocations.
cv2.ocl.setUseOpenCL(False)


def parse_args():
    """Parse in command line arguments"""
    parser = argparse.ArgumentParser(description='Test a Fast R-CNN network')
    parser.add_argument(
        '--dataset',
        help='training dataset')
    parser.add_argument(
        '--cfg', dest='cfg_file', required=True,
        help='optional config file')

    parser.add_argument(
        '--rel_output_dir',
        help='output directory to save the rel testing results.')
    
    parser.add_argument(
        '--att_output_dir',
        help='output directory to save the att testing results.')

    parser.add_argument(
        '--set', dest='set_cfgs',
        help='set config keys, will overwrite config in the cfg_file.'
             ' See lib/core/config.py for all options',
        default=[], nargs='*')

    parser.add_argument(
        '--range',
        help='start (inclusive) and end (exclusive) indices',
        type=int, nargs=2)
    parser.add_argument(
        '--multi-gpu-testing', help='using multiple gpus for inference',
        action='store_true')
    parser.add_argument(
        '--vis', dest='vis', help='visualize detections', action='store_true')
    parser.add_argument(
        '--do_val', dest='do_val', help='do evaluation', action='store_true')

    return parser.parse_args()


if __name__ == '__main__':

    logger = utils.logging.setup_logging(__name__)
    args = parse_args()
    logger.info('Called with args:')
    logger.info(args)

    cfg.VIS = args.vis

    if args.cfg_file is not None:
        merge_cfg_from_file(args.cfg_file)
    if args.set_cfgs is not None:
        merge_cfg_from_list(args.set_cfgs)

    if args.dataset == "oi_all":
        cfg.TEST.DATASETS = ('oi_all_rel_val', 'oi_all_att_val')
    elif args.dataset == "oi_kaggle":
        cfg.TEST.DATASETS = ('oi_kaggle_rel_test', 'oi_kaggle_att_test')
    else:  # For subprocess call
        assert cfg.TEST.DATASETS, 'cfg.TEST.DATASETS shouldn\'t be empty'
    assert_and_infer_cfg()
    
    assert os.path.exists(args.rel_output_dir)

    logger.info('Testing with config:')
    logger.info(pprint.pformat(cfg))
    
    rel_file = os.path.join(args.rel_output_dir, 'rel_detections.pkl')
    att_file = os.path.join(args.att_output_dir, 'att_detections.pkl')
    assert os.path.exists(rel_file) and os.path.exists(att_file)
    logger.info('Loading rel results from {}'.format(rel_file))
    with open(rel_file, 'rb') as f:
        rel_results = pickle.load(f)
    logger.info('Loading att results from {}'.format(att_file))
    with open(att_file, 'rb') as f:
        att_results = pickle.load(f)
    logger.info('Starting evaluation now...')
    task_evaluation_sg_rel_and_att.eval_rel_results(rel_results, att_results, args.rel_output_dir, args.do_val)
