# Adapted by Ji Zhang in 2019
#
# Based on Detectron.pytorch/lib/roi_data/fast_rcnn.py
# Original license text:
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

"""Construct minibatches for Fast R-CNN training. Handles the minibatch blobs
that are specific to Fast R-CNN. Other blobs that are generic to RPN, etc.
are handled by their respecitive roi_data modules.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import numpy as np
import numpy.random as npr
import logging

from core.config import cfg
import utils.blob as blob_utils
import utils.fpn as fpn_utils


logger = logging.getLogger(__name__)


def add_att_blobs(blobs, im_scales, roidb):
    """Add blobs needed for training Fast R-CNN style models."""
    # Sample training RoIs from each image and append them to the blob lists
    for im_i, entry in enumerate(roidb):
        frcn_blobs = _sample_atts(entry, im_scales[im_i], im_i)
        for k, v in frcn_blobs.items():
            blobs[k].append(v)
    # Concat the training blob lists into tensors
    for k, v in blobs.items():
        if isinstance(v, list) and len(v) > 0:
            blobs[k] = np.concatenate(v)
            
    if cfg.FPN.FPN_ON and cfg.FPN.MULTILEVEL_ROIS:
        _add_att_multilevel_rois(blobs)

    return True


def _sample_atts(roidb, im_scale, batch_idx):
    """Generate a random sample of RoIs comprising foreground and background
    examples.
    """
    fg_per_image = cfg.TRAIN.FG_ATT_SIZE_PER_IM
    all_per_image = int(cfg.TRAIN.FG_ATT_SIZE_PER_IM / cfg.TRAIN.FG_ATT_FRACTION)
    max_att_overlaps = roidb['max_att_overlaps']

    gt_inds = np.where(max_att_overlaps > 1.0 - 1e-4)[0]
    fg_inds = np.where((max_att_overlaps >= cfg.TRAIN.FG_THRESH) &
                       (max_att_overlaps <= 1.0 - 1e-4))[0]
    
    fg_per_this_image = np.minimum(fg_per_image, gt_inds.size + fg_inds.size)
    # Sample foreground regions without replacement
    if fg_inds.size > 0 and fg_per_this_image > gt_inds.size:
        fg_inds = npr.choice(
            fg_inds, size=(fg_per_this_image - gt_inds.size), replace=False)
    fg_inds = np.append(fg_inds, gt_inds)

    # Label is the class each RoI has max overlap with
    fg_att_labels = roidb['max_att_classes'][fg_inds]
    blob_dict = dict(
        fg_att_labels_int32=fg_att_labels.astype(np.int32, copy=False))
    if cfg.MODEL.USE_BG:
        bg_inds = np.where((max_att_overlaps < cfg.TRAIN.BG_THRESH_HI))[0]
        
        # Compute number of background RoIs to take from this image (guarding
        # against there being fewer than desired)
        bg_per_this_image = all_per_image - fg_per_this_image
        bg_per_this_image = np.minimum(bg_per_this_image, bg_inds.size)
        # Sample foreground regions without replacement
        if bg_inds.size > 0:
            bg_inds = npr.choice(
                bg_inds, size=bg_per_this_image, replace=False)
        keep_inds = np.append(fg_inds, bg_inds)
        all_att_labels = np.zeros(keep_inds.size, dtype=np.int32)
        all_att_labels[:fg_inds.size] = fg_att_labels + 1  # class should start from 1
        
    else:
        keep_inds = fg_inds
        all_att_labels = fg_att_labels
    blob_dict['all_att_labels_int32'] = all_att_labels.astype(np.int32, copy=False)
    blob_dict['fg_size'] = np.array([fg_inds.size], dtype=np.int32)  # this is used to check if there is at least one fg to learn

    sampled_obj_boxes = roidb['obj_boxes'][keep_inds]
    # Scale rois and format as (batch_idx, x1, y1, x2, y2)
    sampled_obj_rois = sampled_obj_boxes * im_scale
    repeated_batch_idx = batch_idx * blob_utils.ones((keep_inds.shape[0], 1))
    sampled_obj_rois = np.hstack((repeated_batch_idx, sampled_obj_rois))
    blob_dict['obj_rois'] = sampled_obj_rois
    if cfg.MODEL.USE_FREQ_BIAS:
        obj_labels = roidb['max_obj_classes'][keep_inds]
        blob_dict['all_obj_labels_int32'] = obj_labels.astype(np.int32, copy=False)

    return blob_dict


def _add_att_multilevel_rois(blobs):
    """By default training RoIs are added for a single feature map level only.
    When using FPN, the RoIs must be distributed over different FPN levels
    according the level assignment heuristic (see: modeling.FPN.
    map_rois_to_fpn_levels).
    """
    lvl_min = cfg.FPN.ROI_MIN_LEVEL
    lvl_max = cfg.FPN.ROI_MAX_LEVEL

    def _distribute_rois_over_fpn_levels(rois_blob_name):
        """Distribute rois over the different FPN levels."""
        # Get target level for each roi
        # Recall blob rois are in (batch_idx, x1, y1, x2, y2) format, hence take
        # the box coordinates from columns 1:5
        target_lvls = fpn_utils.map_rois_to_fpn_levels(
            blobs[rois_blob_name][:, 1:5], lvl_min, lvl_max
        )
        # Add per FPN level roi blobs named like: <rois_blob_name>_fpn<lvl>
        fpn_utils.add_multilevel_roi_blobs(
            blobs, rois_blob_name, blobs[rois_blob_name], target_lvls, lvl_min,
            lvl_max
        )

    _distribute_rois_over_fpn_levels('obj_rois')
