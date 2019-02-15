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
from core.test_engine_att import run_inference
import utils_rel.logging

from datasets_rel import task_evaluation_sg_att as task_evaluation_sg_att

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
        '--load_ckpt', help='path of checkpoint to load')
    parser.add_argument(
        '--load_detectron', help='path to the detectron weight pickle file')

    parser.add_argument(
        '--output_dir',
        help='output directory to save the testing results. If not provided, '
             'defaults to [args.load_ckpt|args.load_detectron]/../test.')

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

    if not torch.cuda.is_available():
        sys.exit("Need a CUDA device to run the code.")

    logger = utils.logging.setup_logging(__name__)
    args = parse_args()
    logger.info('Called with args:')
    logger.info(args)

    assert (torch.cuda.device_count() == 1) ^ bool(args.multi_gpu_testing)

    cfg.VIS = args.vis

    if args.cfg_file is not None:
        merge_cfg_from_file(args.cfg_file)
    if args.set_cfgs is not None:
        merge_cfg_from_file(args.cfg_file)

    if args.dataset == "oi_att":
        cfg.TEST.DATASETS = ('oi_att_val',)
        cfg.MODEL.NUM_CLASSES = 58
        cfg.MODEL.NUM_ATT_CLASSES = 5  # att, exclude background
    elif args.dataset == "oi_att_mini":
        cfg.TEST.DATASETS = ('oi_att_val_mini',)
        cfg.MODEL.NUM_CLASSES = 58
        cfg.MODEL.NUM_ATT_CLASSES = 5  # att, exclude background
    elif args.dataset == "oi_all_att_train":
        cfg.TEST.DATASETS = ('oi_all_att_train',)
        cfg.MODEL.NUM_CLASSES = 58
        cfg.MODEL.NUM_ATT_CLASSES = 5  # att, exclude background
    elif args.dataset == "oi_all_att":
        cfg.TEST.DATASETS = ('oi_all_att_val',)
        cfg.MODEL.NUM_CLASSES = 58
        cfg.MODEL.NUM_ATT_CLASSES = 5  # att, exclude background
    elif args.dataset == "oi_kaggle":
        cfg.TEST.DATASETS = ('oi_kaggle_att_test',)
        cfg.MODEL.NUM_CLASSES = 58
        cfg.MODEL.NUM_ATT_CLASSES = 5  # att, exclude background
    else:  # For subprocess call
        assert cfg.TEST.DATASETS, 'cfg.TEST.DATASETS shouldn\'t be empty'

    assert_and_infer_cfg()
    
    if not cfg.MODEL.RUN_BASELINE:
        assert bool(args.load_ckpt) ^ bool(args.load_detectron), \
            'Exactly one of --load_ckpt and --load_detectron should be specified.'
    if args.output_dir is None:
        ckpt_path = args.load_ckpt if args.load_ckpt else args.load_detectron
        args.output_dir = os.path.join(
            os.path.dirname(os.path.dirname(ckpt_path)), 'test')
        logger.info('Automatically set output directory to %s', args.output_dir)
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    logger.info('Testing with config:')
    logger.info(pprint.pformat(cfg))

    # For test_engine.multi_gpu_test_net_on_dataset
    args.test_net_file, _ = os.path.splitext(__file__)
    # manually set args.cuda
    args.cuda = True

    det_file = os.path.join(args.output_dir, 'att_detections.pkl')
    if os.path.exists(det_file):
        logger.info('Loading results from {}'.format(det_file))
        with open(det_file, 'rb') as f:
            all_results = pickle.load(f)
        logger.info('Starting evaluation now...')
        task_evaluation_sg_att.eval_att_results(all_results, args.output_dir, args.do_val)
    else:
        run_inference(
            args,
            ind_range=args.range,
            multi_gpu_testing=args.multi_gpu_testing,
            check_expected_results=True)
