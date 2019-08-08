# Adapted by Ji Zhang in 2019
# From Detectron.pytorch/lib/core/test.py
# Original license text below:
# --------------------------------------------------------
# Written by Roy Tseng
#
# Based on:
# --------------------------------------------------------
# Copyright (c) 2017-present, Facebook, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
##############################################################################
#
# Based on:
# --------------------------------------------------------
# Fast R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# --------------------------------------------------------
##############################################################################

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from collections import defaultdict
from six.moves import cPickle as pickle
import cv2
import numpy as np
import pycocotools.mask as mask_util

from torch.autograd import Variable
import torch

from core.config import cfg
from utils.timer import Timer
import utils.blob as blob_utils
import utils.fpn as fpn_utils
import utils.image as image_utils


def im_detect_atts(model, im, dataset_name, box_proposals, timers=None):
    
    if timers is None:
        timers = defaultdict(Timer)
    
    timers['im_detect_atts'].tic()
    att_results = im_get_det_atts(model, im, dataset_name, cfg.TEST.SCALE, cfg.TEST.MAX_SIZE, box_proposals)
    timers['im_detect_atts'].toc()
    
    return att_results


def im_get_det_atts(model, im, dataset_name, target_scale, target_max_size, boxes=None):
    """Prepare the bbox for testing"""

    inputs, im_scale = _get_blobs(im, boxes, target_scale, target_max_size)

    if cfg.DEDUP_BOXES > 0 and not cfg.MODEL.FASTER_RCNN:
        v = np.array([1, 1e3, 1e6, 1e9, 1e12])
        hashes = np.round(inputs['rois'] * cfg.DEDUP_BOXES).dot(v)
        _, index, inv_index = np.unique(
            hashes, return_index=True, return_inverse=True
        )
        inputs['rois'] = inputs['rois'][index, :]
        boxes = boxes[index, :]

    # Add multi-level rois for FPN
    if cfg.FPN.MULTILEVEL_ROIS and not cfg.MODEL.FASTER_RCNN:
        _add_multilevel_rois_for_test(inputs, 'rois')

    if cfg.PYTORCH_VERSION_LESS_THAN_040:
        inputs['data'] = [Variable(torch.from_numpy(inputs['data']), volatile=True)]
        inputs['im_info'] = [Variable(torch.from_numpy(inputs['im_info']), volatile=True)]
    else:
        inputs['data'] = [torch.from_numpy(inputs['data'])]
        inputs['im_info'] = [torch.from_numpy(inputs['im_info'])]
    if dataset_name is not None:
        inputs['dataset_name'] = [blob_utils.serialize(dataset_name)]

    return_dict = model(**inputs)
    
    return_dict2 = {}
    if return_dict['obj_rois'] is not None:
        obj_boxes = return_dict['obj_rois'].data.cpu().numpy()[:, 1:5] / im_scale
        obj_labels = return_dict['obj_labels'].data.cpu().numpy() - 1
        obj_scores = return_dict['obj_scores'].data.cpu().numpy()
        att_scores = return_dict['att_scores'].data.cpu().numpy()
        return_dict2['obj_boxes'] = obj_boxes
        return_dict2['obj_labels'] = obj_labels.astype(np.int32, copy=False)
        return_dict2['obj_scores'] = obj_scores
        return_dict2['att_scores'] = att_scores
    else:
        return_dict2['obj_boxes'] = None
        return_dict2['obj_labels'] = None
        return_dict2['obj_scores'] = None
        return_dict2['att_scores'] = None
    
    return return_dict2


def _get_rois_blob(im_rois, im_scale):
    """Converts RoIs into network inputs.

    Arguments:
        im_rois (ndarray): R x 4 matrix of RoIs in original image coordinates
        im_scale_factors (list): scale factors as returned by _get_image_blob

    Returns:
        blob (ndarray): R x 5 matrix of RoIs in the image pyramid with columns
            [level, x1, y1, x2, y2]
    """
    rois, levels = _project_im_rois(im_rois, im_scale)
    rois_blob = np.hstack((levels, rois))
    return rois_blob.astype(np.float32, copy=False)


def _project_im_rois(im_rois, scales):
    """Project image RoIs into the image pyramid built by _get_image_blob.

    Arguments:
        im_rois (ndarray): R x 4 matrix of RoIs in original image coordinates
        scales (list): scale factors as returned by _get_image_blob

    Returns:
        rois (ndarray): R x 4 matrix of projected RoI coordinates
        levels (ndarray): image pyramid levels used by each projected RoI
    """
    rois = im_rois.astype(np.float, copy=False) * scales
    levels = np.zeros((im_rois.shape[0], 1), dtype=np.int)
    return rois, levels


def _add_multilevel_rois_for_test(blobs, name):
    """Distributes a set of RoIs across FPN pyramid levels by creating new level
    specific RoI blobs.

    Arguments:
        blobs (dict): dictionary of blobs
        name (str): a key in 'blobs' identifying the source RoI blob

    Returns:
        [by ref] blobs (dict): new keys named by `name + 'fpn' + level`
            are added to dict each with a value that's an R_level x 5 ndarray of
            RoIs (see _get_rois_blob for format)
    """
    lvl_min = cfg.FPN.ROI_MIN_LEVEL
    lvl_max = cfg.FPN.ROI_MAX_LEVEL
    lvls = fpn_utils.map_rois_to_fpn_levels(blobs[name][:, 1:5], lvl_min, lvl_max)
    fpn_utils.add_multilevel_roi_blobs(
        blobs, name, blobs[name], lvls, lvl_min, lvl_max
    )


def _get_blobs(im, rois, target_scale, target_max_size):
    """Convert an image and RoIs within that image into network inputs."""
    blobs = {}
    blobs['data'], im_scale, blobs['im_info'] = \
        blob_utils.get_image_blob(im, target_scale, target_max_size)
    if rois is not None:
        blobs['rois'] = _get_rois_blob(rois, im_scale)
    return blobs, im_scale
