# Adapted by Ji Zhang, 2019
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
import utils_rel.boxes_rel as box_utils_rel
import utils.blob as blob_utils
import utils.fpn as fpn_utils


logger = logging.getLogger(__name__)


def add_rel_blobs(blobs, im_scales, roidb):
    """Add blobs needed for training Fast R-CNN style models."""
    # Sample training RoIs from each image and append them to the blob lists
    for im_i, entry in enumerate(roidb):
        frcn_blobs = _sample_pairs(entry, im_scales[im_i], im_i)
        for k, v in frcn_blobs.items():
            blobs[k].append(v)
    # Concat the training blob lists into tensors
    for k, v in blobs.items():
        if isinstance(v, list) and len(v) > 0:
            blobs[k] = np.concatenate(v)
            
    if cfg.FPN.FPN_ON and cfg.FPN.MULTILEVEL_ROIS:
        _add_rel_multilevel_rois(blobs)

    return True


def _sample_pairs(roidb, im_scale, batch_idx):
    """Generate a random sample of RoIs comprising foreground and background
    examples.
    """
    fg_pairs_per_image = cfg.TRAIN.FG_REL_SIZE_PER_IM
    pairs_per_image = int(cfg.TRAIN.FG_REL_SIZE_PER_IM / cfg.TRAIN.FG_REL_FRACTION)  # need much more pairs since it's quadratic
    max_pair_overlaps = roidb['max_pair_overlaps']

    gt_pair_inds = np.where(max_pair_overlaps > 1.0 - 1e-4)[0]
    fg_pair_inds = np.where((max_pair_overlaps >= cfg.TRAIN.FG_THRESH) &
                            (max_pair_overlaps <= 1.0 - 1e-4))[0]
    
    fg_pairs_per_this_image = np.minimum(fg_pairs_per_image, gt_pair_inds.size + fg_pair_inds.size)
    # Sample foreground regions without replacement
    if fg_pair_inds.size > 0:
        fg_pair_inds = npr.choice(
            fg_pair_inds, size=(fg_pairs_per_this_image - gt_pair_inds.size), replace=False)
    fg_pair_inds = np.append(fg_pair_inds, gt_pair_inds)

    # Label is the class each RoI has max overlap with
    fg_prd_labels = roidb['max_prd_classes'][fg_pair_inds]
    blob_dict = dict(
        fg_prd_labels_int32=fg_prd_labels.astype(np.int32, copy=False))
    if cfg.MODEL.USE_BG:
        bg_pair_inds = np.where((max_pair_overlaps < cfg.TRAIN.BG_THRESH_HI))[0]
        
        # Compute number of background RoIs to take from this image (guarding
        # against there being fewer than desired)
        bg_pairs_per_this_image = pairs_per_image - fg_pairs_per_this_image
        bg_pairs_per_this_image = np.minimum(bg_pairs_per_this_image, bg_pair_inds.size)
        # Sample foreground regions without replacement
        if bg_pair_inds.size > 0:
            bg_pair_inds = npr.choice(
                bg_pair_inds, size=bg_pairs_per_this_image, replace=False)
        keep_pair_inds = np.append(fg_pair_inds, bg_pair_inds)
        all_prd_labels = np.zeros(keep_pair_inds.size, dtype=np.int32)
        all_prd_labels[:fg_pair_inds.size] = fg_prd_labels + 1  # class should start from 1
    else:
        keep_pair_inds = fg_pair_inds
        all_prd_labels = fg_prd_labels
    blob_dict['all_prd_labels_int32'] = all_prd_labels.astype(np.int32, copy=False)
    blob_dict['fg_size'] = np.array([fg_pair_inds.size], dtype=np.int32)  # this is used to check if there is at least one fg to learn

    sampled_sbj_boxes = roidb['sbj_boxes'][keep_pair_inds]
    sampled_obj_boxes = roidb['obj_boxes'][keep_pair_inds]
    # Scale rois and format as (batch_idx, x1, y1, x2, y2)
    sampled_sbj_rois = sampled_sbj_boxes * im_scale
    sampled_obj_rois = sampled_obj_boxes * im_scale
    repeated_batch_idx = batch_idx * blob_utils.ones((keep_pair_inds.shape[0], 1))
    sampled_sbj_rois = np.hstack((repeated_batch_idx, sampled_sbj_rois))
    sampled_obj_rois = np.hstack((repeated_batch_idx, sampled_obj_rois))
    blob_dict['sbj_rois'] = sampled_sbj_rois
    blob_dict['obj_rois'] = sampled_obj_rois
    sampled_rel_rois = box_utils_rel.rois_union(sampled_sbj_rois, sampled_obj_rois)
    blob_dict['rel_rois'] = sampled_rel_rois
    if cfg.MODEL.USE_SPATIAL_FEAT:
        sampled_spt_feat = box_utils_rel.get_spt_features(
            sampled_sbj_boxes, sampled_obj_boxes, roidb['width'], roidb['height'])
        blob_dict['spt_feat'] = sampled_spt_feat
    if cfg.MODEL.USE_FREQ_BIAS:
        sbj_labels = roidb['max_sbj_classes'][keep_pair_inds]
        obj_labels = roidb['max_obj_classes'][keep_pair_inds]
        blob_dict['all_sbj_labels_int32'] = sbj_labels.astype(np.int32, copy=False)
        blob_dict['all_obj_labels_int32'] = obj_labels.astype(np.int32, copy=False)
    if cfg.MODEL.USE_NODE_CONTRASTIVE_LOSS or cfg.MODEL.USE_NODE_CONTRASTIVE_SO_AWARE_LOSS or cfg.MODEL.USE_NODE_CONTRASTIVE_P_AWARE_LOSS:
        nodes_per_image = cfg.MODEL.NODE_SAMPLE_SIZE
        max_sbj_overlaps = roidb['max_sbj_overlaps']
        max_obj_overlaps = roidb['max_obj_overlaps']
        # sbj
        # Here a naturally existing assumption is, each positive sbj should have at least one positive obj
        sbj_pos_pair_pos_inds = np.where((max_pair_overlaps >= cfg.TRAIN.FG_THRESH))[0]
        sbj_pos_obj_pos_pair_neg_inds = np.where((max_sbj_overlaps >= cfg.TRAIN.FG_THRESH) &
                                                 (max_obj_overlaps >= cfg.TRAIN.FG_THRESH) &
                                                 (max_pair_overlaps < cfg.TRAIN.BG_THRESH_HI))[0]
        sbj_pos_obj_neg_pair_neg_inds = np.where((max_sbj_overlaps >= cfg.TRAIN.FG_THRESH) &
                                                 (max_obj_overlaps < cfg.TRAIN.FG_THRESH) &
                                                 (max_pair_overlaps < cfg.TRAIN.BG_THRESH_HI))[0]
        if sbj_pos_pair_pos_inds.size > 0:
            sbj_pos_pair_pos_inds = npr.choice(
                sbj_pos_pair_pos_inds,
                size=int(min(nodes_per_image, sbj_pos_pair_pos_inds.size)),
                replace=False)
        if sbj_pos_obj_pos_pair_neg_inds.size > 0:
            sbj_pos_obj_pos_pair_neg_inds = npr.choice(
                sbj_pos_obj_pos_pair_neg_inds,
                size=int(min(nodes_per_image, sbj_pos_obj_pos_pair_neg_inds.size)),
                replace=False)
        sbj_pos_pair_neg_inds = sbj_pos_obj_pos_pair_neg_inds
        if nodes_per_image - sbj_pos_obj_pos_pair_neg_inds.size > 0 and sbj_pos_obj_neg_pair_neg_inds.size > 0:
            sbj_pos_obj_neg_pair_neg_inds = npr.choice(
                sbj_pos_obj_neg_pair_neg_inds,
                size=int(min(nodes_per_image - sbj_pos_obj_pos_pair_neg_inds.size, sbj_pos_obj_neg_pair_neg_inds.size)),
                replace=False)
            sbj_pos_pair_neg_inds = np.append(sbj_pos_pair_neg_inds, sbj_pos_obj_neg_pair_neg_inds)
        sbj_pos_inds = np.append(sbj_pos_pair_pos_inds, sbj_pos_pair_neg_inds)
        binary_labels_sbj_pos = np.zeros(sbj_pos_inds.size, dtype=np.int32)
        binary_labels_sbj_pos[:sbj_pos_pair_pos_inds.size] = 1
        blob_dict['binary_labels_sbj_pos_int32'] = binary_labels_sbj_pos.astype(np.int32, copy=False)
        prd_pos_labels_sbj_pos = roidb['max_prd_classes'][sbj_pos_pair_pos_inds]
        prd_labels_sbj_pos = np.zeros(sbj_pos_inds.size, dtype=np.int32)
        prd_labels_sbj_pos[:sbj_pos_pair_pos_inds.size] = prd_pos_labels_sbj_pos + 1
        blob_dict['prd_labels_sbj_pos_int32'] = prd_labels_sbj_pos.astype(np.int32, copy=False)
        sbj_labels_sbj_pos = roidb['max_sbj_classes'][sbj_pos_inds] + 1
        # 1. set all obj labels > 0
        obj_labels_sbj_pos = roidb['max_obj_classes'][sbj_pos_inds] + 1
        # 2. find those negative obj
        max_obj_overlaps_sbj_pos = roidb['max_obj_overlaps'][sbj_pos_inds]
        obj_neg_inds_sbj_pos = np.where(max_obj_overlaps_sbj_pos < cfg.TRAIN.FG_THRESH)[0]
        obj_labels_sbj_pos[obj_neg_inds_sbj_pos] = 0
        blob_dict['sbj_labels_sbj_pos_int32'] = sbj_labels_sbj_pos.astype(np.int32, copy=False)
        blob_dict['obj_labels_sbj_pos_int32'] = obj_labels_sbj_pos.astype(np.int32, copy=False)
        # this is for freq bias in RelDN
        blob_dict['sbj_labels_sbj_pos_fg_int32'] = roidb['max_sbj_classes'][sbj_pos_inds].astype(np.int32, copy=False)
        blob_dict['obj_labels_sbj_pos_fg_int32'] = roidb['max_obj_classes'][sbj_pos_inds].astype(np.int32, copy=False)
        
        sampled_sbj_boxes_sbj_pos = roidb['sbj_boxes'][sbj_pos_inds]
        sampled_obj_boxes_sbj_pos = roidb['obj_boxes'][sbj_pos_inds]
        # Scale rois and format as (batch_idx, x1, y1, x2, y2)
        sampled_sbj_rois_sbj_pos = sampled_sbj_boxes_sbj_pos * im_scale
        sampled_obj_rois_sbj_pos = sampled_obj_boxes_sbj_pos * im_scale
        repeated_batch_idx = batch_idx * blob_utils.ones((sbj_pos_inds.shape[0], 1))
        sampled_sbj_rois_sbj_pos = np.hstack((repeated_batch_idx, sampled_sbj_rois_sbj_pos))
        sampled_obj_rois_sbj_pos = np.hstack((repeated_batch_idx, sampled_obj_rois_sbj_pos))
        blob_dict['sbj_rois_sbj_pos'] = sampled_sbj_rois_sbj_pos
        blob_dict['obj_rois_sbj_pos'] = sampled_obj_rois_sbj_pos
        sampled_rel_rois_sbj_pos = box_utils_rel.rois_union(sampled_sbj_rois_sbj_pos, sampled_obj_rois_sbj_pos)
        blob_dict['rel_rois_sbj_pos'] = sampled_rel_rois_sbj_pos
        _, inds_unique_sbj_pos, inds_reverse_sbj_pos = np.unique(
            sampled_sbj_rois_sbj_pos, return_index=True, return_inverse=True, axis=0)
        assert inds_reverse_sbj_pos.shape[0] == sampled_sbj_rois_sbj_pos.shape[0]
        blob_dict['inds_unique_sbj_pos'] = inds_unique_sbj_pos
        blob_dict['inds_reverse_sbj_pos'] = inds_reverse_sbj_pos
        if cfg.MODEL.USE_SPATIAL_FEAT:
            sampled_spt_feat_sbj_pos = box_utils_rel.get_spt_features(
                sampled_sbj_boxes_sbj_pos, sampled_obj_boxes_sbj_pos, roidb['width'], roidb['height'])
            blob_dict['spt_feat_sbj_pos'] = sampled_spt_feat_sbj_pos
        # obj
        # Here a naturally existing assumption is, each positive obj should have at least one positive sbj
        obj_pos_pair_pos_inds = np.where((max_pair_overlaps >= cfg.TRAIN.FG_THRESH))[0]
        obj_pos_sbj_pos_pair_neg_inds = np.where((max_obj_overlaps >= cfg.TRAIN.FG_THRESH) &
                                                 (max_sbj_overlaps >= cfg.TRAIN.FG_THRESH) &
                                                 (max_pair_overlaps < cfg.TRAIN.BG_THRESH_HI))[0]
        obj_pos_sbj_neg_pair_neg_inds = np.where((max_obj_overlaps >= cfg.TRAIN.FG_THRESH) &
                                                 (max_sbj_overlaps < cfg.TRAIN.FG_THRESH) &
                                                 (max_pair_overlaps < cfg.TRAIN.BG_THRESH_HI))[0]
        if obj_pos_pair_pos_inds.size > 0:
            obj_pos_pair_pos_inds = npr.choice(
                obj_pos_pair_pos_inds,
                size=int(min(nodes_per_image, obj_pos_pair_pos_inds.size)),
                replace=False)
        if obj_pos_sbj_pos_pair_neg_inds.size > 0:
            obj_pos_sbj_pos_pair_neg_inds = npr.choice(
                obj_pos_sbj_pos_pair_neg_inds,
                size=int(min(nodes_per_image, obj_pos_sbj_pos_pair_neg_inds.size)),
                replace=False)
        obj_pos_pair_neg_inds = obj_pos_sbj_pos_pair_neg_inds
        if nodes_per_image - obj_pos_sbj_pos_pair_neg_inds.size > 0 and obj_pos_sbj_neg_pair_neg_inds.size:
            obj_pos_sbj_neg_pair_neg_inds = npr.choice(
                obj_pos_sbj_neg_pair_neg_inds,
                size=int(min(nodes_per_image - obj_pos_sbj_pos_pair_neg_inds.size, obj_pos_sbj_neg_pair_neg_inds.size)),
                replace=False)
            obj_pos_pair_neg_inds = np.append(obj_pos_pair_neg_inds, obj_pos_sbj_neg_pair_neg_inds)
        obj_pos_inds = np.append(obj_pos_pair_pos_inds, obj_pos_pair_neg_inds)
        binary_labels_obj_pos = np.zeros(obj_pos_inds.size, dtype=np.int32)
        binary_labels_obj_pos[:obj_pos_pair_pos_inds.size] = 1
        blob_dict['binary_labels_obj_pos_int32'] = binary_labels_obj_pos.astype(np.int32, copy=False)
        prd_pos_labels_obj_pos = roidb['max_prd_classes'][obj_pos_pair_pos_inds]
        prd_labels_obj_pos = np.zeros(obj_pos_inds.size, dtype=np.int32)
        prd_labels_obj_pos[:obj_pos_pair_pos_inds.size] = prd_pos_labels_obj_pos + 1
        blob_dict['prd_labels_obj_pos_int32'] = prd_labels_obj_pos.astype(np.int32, copy=False)
        obj_labels_obj_pos = roidb['max_obj_classes'][obj_pos_inds] + 1
        # 1. set all sbj labels > 0
        sbj_labels_obj_pos = roidb['max_sbj_classes'][obj_pos_inds] + 1
        # 2. find those negative sbj
        max_sbj_overlaps_obj_pos = roidb['max_sbj_overlaps'][obj_pos_inds]
        sbj_neg_inds_obj_pos = np.where(max_sbj_overlaps_obj_pos < cfg.TRAIN.FG_THRESH)[0]
        sbj_labels_obj_pos[sbj_neg_inds_obj_pos] = 0
        blob_dict['sbj_labels_obj_pos_int32'] = sbj_labels_obj_pos.astype(np.int32, copy=False)
        blob_dict['obj_labels_obj_pos_int32'] = obj_labels_obj_pos.astype(np.int32, copy=False)
        # this is for freq bias in RelDN
        blob_dict['sbj_labels_obj_pos_fg_int32'] = roidb['max_sbj_classes'][obj_pos_inds].astype(np.int32, copy=False)
        blob_dict['obj_labels_obj_pos_fg_int32'] = roidb['max_obj_classes'][obj_pos_inds].astype(np.int32, copy=False)
        
        sampled_sbj_boxes_obj_pos = roidb['sbj_boxes'][obj_pos_inds]
        sampled_obj_boxes_obj_pos = roidb['obj_boxes'][obj_pos_inds]
        # Scale rois and format as (batch_idx, x1, y1, x2, y2)
        sampled_sbj_rois_obj_pos = sampled_sbj_boxes_obj_pos * im_scale
        sampled_obj_rois_obj_pos = sampled_obj_boxes_obj_pos * im_scale
        repeated_batch_idx = batch_idx * blob_utils.ones((obj_pos_inds.shape[0], 1))
        sampled_sbj_rois_obj_pos = np.hstack((repeated_batch_idx, sampled_sbj_rois_obj_pos))
        sampled_obj_rois_obj_pos = np.hstack((repeated_batch_idx, sampled_obj_rois_obj_pos))
        blob_dict['sbj_rois_obj_pos'] = sampled_sbj_rois_obj_pos
        blob_dict['obj_rois_obj_pos'] = sampled_obj_rois_obj_pos
        sampled_rel_rois_obj_pos = box_utils_rel.rois_union(sampled_sbj_rois_obj_pos, sampled_obj_rois_obj_pos)
        blob_dict['rel_rois_obj_pos'] = sampled_rel_rois_obj_pos
        _, inds_unique_obj_pos, inds_reverse_obj_pos = np.unique(
            sampled_obj_rois_obj_pos, return_index=True, return_inverse=True, axis=0)
        assert inds_reverse_obj_pos.shape[0] == sampled_obj_rois_obj_pos.shape[0]
        blob_dict['inds_unique_obj_pos'] = inds_unique_obj_pos
        blob_dict['inds_reverse_obj_pos'] = inds_reverse_obj_pos
        if cfg.MODEL.USE_SPATIAL_FEAT:
            sampled_spt_feat_obj_pos = box_utils_rel.get_spt_features(
                sampled_sbj_boxes_obj_pos, sampled_obj_boxes_obj_pos, roidb['width'], roidb['height'])
            blob_dict['spt_feat_obj_pos'] = sampled_spt_feat_obj_pos

    return blob_dict


def _add_rel_multilevel_rois(blobs):
    """By default training RoIs are added for a single feature map level only.
    When using FPN, the RoIs must be distributed over different FPN levels
    according the level assignment heuristic (see: modeling.FPN.
    map_rois_to_fpn_levels).
    """
    lvl_min = cfg.FPN.ROI_MIN_LEVEL
    lvl_max = cfg.FPN.ROI_MAX_LEVEL

    def _distribute_rois_over_fpn_levels(rois_blob_names):
        """Distribute rois over the different FPN levels."""
        # Get target level for each roi
        # Recall blob rois are in (batch_idx, x1, y1, x2, y2) format, hence take
        # the box coordinates from columns 1:5
        lowest_target_lvls = None
        for rois_blob_name in rois_blob_names:
            target_lvls = fpn_utils.map_rois_to_fpn_levels(
                blobs[rois_blob_name][:, 1:5], lvl_min, lvl_max)
            if lowest_target_lvls is None:
                lowest_target_lvls = target_lvls
            else:
                lowest_target_lvls = np.minimum(lowest_target_lvls, target_lvls)
        for rois_blob_name in rois_blob_names:
            # Add per FPN level roi blobs named like: <rois_blob_name>_fpn<lvl>
            fpn_utils.add_multilevel_roi_blobs(
                blobs, rois_blob_name, blobs[rois_blob_name], lowest_target_lvls, lvl_min,
                lvl_max)

    _distribute_rois_over_fpn_levels(['sbj_rois'])
    _distribute_rois_over_fpn_levels(['obj_rois'])
    _distribute_rois_over_fpn_levels(['rel_rois'])
    if cfg.MODEL.USE_NODE_CONTRASTIVE_LOSS or cfg.MODEL.USE_NODE_CONTRASTIVE_SO_AWARE_LOSS or cfg.MODEL.USE_NODE_CONTRASTIVE_P_AWARE_LOSS:
        _distribute_rois_over_fpn_levels(['sbj_rois_sbj_pos'])
        _distribute_rois_over_fpn_levels(['obj_rois_sbj_pos'])
        _distribute_rois_over_fpn_levels(['rel_rois_sbj_pos'])
        _distribute_rois_over_fpn_levels(['sbj_rois_obj_pos'])
        _distribute_rois_over_fpn_levels(['obj_rois_obj_pos'])
        _distribute_rois_over_fpn_levels(['rel_rois_obj_pos'])
