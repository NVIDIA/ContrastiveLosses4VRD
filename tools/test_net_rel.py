# Adapted by Ji Zhang, 2019
#
# Based on Detectron.pytorch/tools/test_net.py Written by Roy Tseng

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
import utils.logging

from datasets_rel import task_evaluation_sg as task_evaluation_sg
from datasets_rel import task_evaluation_vg_and_vrd as task_evaluation_vg_and_vrd

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
        '--topk', dest='topk', help='do evaluation', type=int, default=100)
    parser.add_argument(
        '--do_val', dest='do_val', help='do evaluation', action='store_true')
    parser.add_argument(
        '--do_vis', dest='do_vis', help='visualize the last layer of conv_body', action='store_true')
    parser.add_argument(
        '--do_special', dest='do_special', help='visualize the last layer of conv_body', action='store_true')
    parser.add_argument(
        '--use_gt_boxes', dest='use_gt_boxes', help='use gt boxes for sgcls/prdcls', action='store_true')
    parser.add_argument(
        '--use_gt_labels', dest='use_gt_labels', help='use gt boxes for sgcls/prdcls', action='store_true')

    return parser.parse_args()


if __name__ == '__main__':
    
    if not torch.cuda.is_available():
        sys.exit("Need a CUDA device to run the code.")

    logger = utils.logging.setup_logging(__name__)
    args = parse_args()
    logger.info('Called with args:')
    logger.info(args)
    
    assert (torch.cuda.device_count() == 1) ^ bool(args.multi_gpu_testing)

    if args.cfg_file is not None:
        merge_cfg_from_file(args.cfg_file)
    if args.set_cfgs is not None:
        merge_cfg_from_file(args.cfg_file)
        
    if args.dataset == "oi_rel":
        cfg.TEST.DATASETS = ('oi_rel_val',)
        cfg.MODEL.NUM_CLASSES = 58
        cfg.MODEL.NUM_PRD_CLASSES = 9  # rel, exclude background
    elif args.dataset == "oi_rel_mini":
        cfg.TEST.DATASETS = ('oi_rel_val_mini',)
        cfg.MODEL.NUM_CLASSES = 58
        cfg.MODEL.NUM_PRD_CLASSES = 9  # rel, exclude background
    elif args.dataset == "oi_all_rel_train":
        cfg.TEST.DATASETS = ('oi_all_rel_train',)
        cfg.MODEL.NUM_CLASSES = 58
        cfg.MODEL.NUM_PRD_CLASSES = 9  # rel, exclude background
    elif args.dataset == "oi_all_rel":
        cfg.TEST.DATASETS = ('oi_all_rel_val',)
        cfg.MODEL.NUM_CLASSES = 58
        cfg.MODEL.NUM_PRD_CLASSES = 9  # rel, exclude background
    elif args.dataset == "oi_kaggle":
        cfg.TEST.DATASETS = ('oi_kaggle_rel_test',)
        cfg.MODEL.NUM_CLASSES = 58
        cfg.MODEL.NUM_PRD_CLASSES = 9  # rel, exclude background
    elif args.dataset == "vg_mini":
        cfg.TEST.DATASETS = ('vg_val_mini',)
        cfg.MODEL.NUM_CLASSES = 151
        cfg.MODEL.NUM_PRD_CLASSES = 50  # exclude background
    elif args.dataset == "vg":
        cfg.TEST.DATASETS = ('vg_val',)
        cfg.MODEL.NUM_CLASSES = 151
        cfg.MODEL.NUM_PRD_CLASSES = 50  # exclude background
    elif args.dataset == "vrd_train":
        cfg.TEST.DATASETS = ('vrd_train',)
        cfg.MODEL.NUM_CLASSES = 101
        cfg.MODEL.NUM_PRD_CLASSES = 70  # exclude background
    elif args.dataset == "vrd":
        cfg.TEST.DATASETS = ('vrd_val',)
        cfg.MODEL.NUM_CLASSES = 101
        cfg.MODEL.NUM_PRD_CLASSES = 70  # exclude background
    elif args.dataset == "gqa":
        cfg.TEST.DATASETS = ('gqa_val',)
        cfg.MODEL.NUM_CLASSES = 1704
        cfg.MODEL.NUM_PRD_CLASSES = 310  # rel, exclude background

    elif args.dataset == "gqa_all":
        cfg.TEST.DATASETS = ('gqa_all',)
        cfg.MODEL.NUM_CLASSES = 1704
        cfg.MODEL.NUM_PRD_CLASSES = 310  # rel, exclude background
    elif args.dataset == "gqa_1st_of_3":
        cfg.TEST.DATASETS = ('gqa_1st_of_3',)
        cfg.MODEL.NUM_CLASSES = 1704
        cfg.MODEL.NUM_PRD_CLASSES = 310  # rel, exclude background
    elif args.dataset == "gqa_2nd_of_3":
        cfg.TEST.DATASETS = ('gqa_2nd_of_3',)
        cfg.MODEL.NUM_CLASSES = 1704
        cfg.MODEL.NUM_PRD_CLASSES = 310  # rel, exclude background
    elif args.dataset == "gqa_3rd_of_3":
        cfg.TEST.DATASETS = ('gqa_3rd_of_3',)
        cfg.MODEL.NUM_CLASSES = 1704
        cfg.MODEL.NUM_PRD_CLASSES = 310  # rel, exclude background

    elif args.dataset == "gqa_spt":
        cfg.TEST.DATASETS = ('gqa_spt_val',)
        cfg.MODEL.NUM_CLASSES = 1321
        cfg.MODEL.NUM_PRD_CLASSES = 23  # rel, exclude background
    elif args.dataset == "gqa_verb":
        cfg.TEST.DATASETS = ('gqa_verb_val',)
        cfg.MODEL.NUM_CLASSES = 1321
        cfg.MODEL.NUM_PRD_CLASSES = 216  # rel, exclude background
    elif args.dataset == "gqa_misc":
        cfg.TEST.DATASETS = ('gqa_misc_val',)
        cfg.MODEL.NUM_CLASSES = 1321
        cfg.MODEL.NUM_PRD_CLASSES = 70  # rel, exclude background

    # rel_spt
    elif args.dataset == "gqa_spt_1st_of_3":
        cfg.TEST.DATASETS = ('gqa_spt_1st_of_3',)
        cfg.MODEL.NUM_CLASSES = 1321
        cfg.MODEL.NUM_PRD_CLASSES = 23  # rel, exclude background
    elif args.dataset == "gqa_spt_2nd_of_3":
        cfg.TEST.DATASETS = ('gqa_spt_2nd_of_3',)
        cfg.MODEL.NUM_CLASSES = 1321
        cfg.MODEL.NUM_PRD_CLASSES = 23  # rel, exclude background
    elif args.dataset == "gqa_spt_3rd_of_3":
        cfg.TEST.DATASETS = ('gqa_spt_3rd_of_3',)
        cfg.MODEL.NUM_CLASSES = 1321
        cfg.MODEL.NUM_PRD_CLASSES = 23  # rel, exclude background
    # rel_verb
    elif args.dataset == "gqa_verb_all":
        cfg.TEST.DATASETS = ('gqa_verb_all',)
        cfg.MODEL.NUM_CLASSES = 1321
        cfg.MODEL.NUM_PRD_CLASSES = 216  # rel, exclude background
    elif args.dataset == "gqa_verb_1st_of_3":
        cfg.TEST.DATASETS = ('gqa_verb_1st_of_3',)
        cfg.MODEL.NUM_CLASSES = 1321
        cfg.MODEL.NUM_PRD_CLASSES = 216  # rel, exclude background
    elif args.dataset == "gqa_verb_2nd_of_3":
        cfg.TEST.DATASETS = ('gqa_verb_2nd_of_3',)
        cfg.MODEL.NUM_CLASSES = 1321
        cfg.MODEL.NUM_PRD_CLASSES = 216  # rel, exclude background
    elif args.dataset == "gqa_verb_3rd_of_3":
        cfg.TEST.DATASETS = ('gqa_verb_3rd_of_3',)
        cfg.MODEL.NUM_CLASSES = 1321
        cfg.MODEL.NUM_PRD_CLASSES = 216  # rel, exclude background
    elif args.dataset == "gqa_verb_1st_of_6":
        cfg.TEST.DATASETS = ('gqa_verb_1st_of_6',)
        cfg.MODEL.NUM_CLASSES = 1321
        cfg.MODEL.NUM_PRD_CLASSES = 216  # rel, exclude background
    elif args.dataset == "gqa_verb_2nd_of_6":
        cfg.TEST.DATASETS = ('gqa_verb_2nd_of_6',)
        cfg.MODEL.NUM_CLASSES = 1321
        cfg.MODEL.NUM_PRD_CLASSES = 216  # rel, exclude background
    elif args.dataset == "gqa_verb_3rd_of_6":
        cfg.TEST.DATASETS = ('gqa_verb_3rd_of_6',)
        cfg.MODEL.NUM_CLASSES = 1321
        cfg.MODEL.NUM_PRD_CLASSES = 216  # rel, exclude background
    elif args.dataset == "gqa_verb_4th_of_6":
        cfg.TEST.DATASETS = ('gqa_verb_4th_of_6',)
        cfg.MODEL.NUM_CLASSES = 1321
        cfg.MODEL.NUM_PRD_CLASSES = 216  # rel, exclude background
    elif args.dataset == "gqa_verb_5th_of_6":
        cfg.TEST.DATASETS = ('gqa_verb_5th_of_6',)
        cfg.MODEL.NUM_CLASSES = 1321
        cfg.MODEL.NUM_PRD_CLASSES = 216  # rel, exclude background
    elif args.dataset == "gqa_verb_6th_of_6":
        cfg.TEST.DATASETS = ('gqa_verb_6th_of_6',)
        cfg.MODEL.NUM_CLASSES = 1321
        cfg.MODEL.NUM_PRD_CLASSES = 216  # rel, exclude background
    # rel_misc
    elif args.dataset == "gqa_misc_1st_of_6":
        cfg.TEST.DATASETS = ('gqa_misc_1st_of_6',)
        cfg.MODEL.NUM_CLASSES = 1321
        cfg.MODEL.NUM_PRD_CLASSES = 70  # rel, exclude background
    elif args.dataset == "gqa_misc_2nd_of_6":
        cfg.TEST.DATASETS = ('gqa_misc_2nd_of_6',)
        cfg.MODEL.NUM_CLASSES = 1321
        cfg.MODEL.NUM_PRD_CLASSES = 70  # rel, exclude background
    elif args.dataset == "gqa_misc_3rd_of_6":
        cfg.TEST.DATASETS = ('gqa_misc_3rd_of_6',)
        cfg.MODEL.NUM_CLASSES = 1321
        cfg.MODEL.NUM_PRD_CLASSES = 70  # rel, exclude background
    elif args.dataset == "gqa_misc_4th_of_6":
        cfg.TEST.DATASETS = ('gqa_misc_4th_of_6',)
        cfg.MODEL.NUM_CLASSES = 1321
        cfg.MODEL.NUM_PRD_CLASSES = 70  # rel, exclude background
    elif args.dataset == "gqa_misc_5th_of_6":
        cfg.TEST.DATASETS = ('gqa_misc_5th_of_6',)
        cfg.MODEL.NUM_CLASSES = 1321
        cfg.MODEL.NUM_PRD_CLASSES = 70  # rel, exclude background
    elif args.dataset == "gqa_misc_6th_of_6":
        cfg.TEST.DATASETS = ('gqa_misc_6th_of_6',)
        cfg.MODEL.NUM_CLASSES = 1321
        cfg.MODEL.NUM_PRD_CLASSES = 70  # rel, exclude background
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

    if args.use_gt_boxes:
        if args.use_gt_labels:
            det_file = os.path.join(args.output_dir, 'rel_detections_gt_boxes_prdcls.pkl')
        else:
            det_file = os.path.join(args.output_dir, 'rel_detections_gt_boxes_sgcls.pkl')
    else:
        det_file = os.path.join(args.output_dir, 'rel_detections.pkl')
    if os.path.exists(det_file):
        logger.info('Loading results from {}'.format(det_file))
        with open(det_file, 'rb') as f:
            all_results = pickle.load(f)
        logger.info('Starting evaluation now...')
        if args.dataset.find('vg') >= 0 or args.dataset.find('vrd') >= 0:
            task_evaluation_vg_and_vrd.eval_rel_results(all_results, args.output_dir, args.topk, args.do_val)
        else:
            task_evaluation_sg.eval_rel_results(all_results, args.output_dir, args.topk, args.do_val, args.do_vis, args.do_special)
    else:
        run_inference(
            args,
            ind_range=args.range,
            multi_gpu_testing=args.multi_gpu_testing,
            check_expected_results=True)
