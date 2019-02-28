# Written by Ji Zhang in 2019

import numpy as np
from numpy import linalg as la
import json
import logging

from torch import nn
from torch.nn import init
import torch.nn.functional as F

from core.config import cfg
from modeling_rel.generate_rel_proposal_labels import GenerateRelProposalLabelsOp
import modeling.FPN as FPN
import utils_rel.boxes_rel as box_utils_rel
import utils.fpn as fpn_utils


logger = logging.getLogger(__name__)


def generic_relpn_outputs():
    return single_scale_relpn_outputs()


class single_scale_relpn_outputs(nn.Module):
    """Add RelPN outputs to a single scale model (i.e., no FPN)."""
    def __init__(self):
        super().__init__()
        
        self.RelPN_GenerateProposalLabels = GenerateRelProposalLabelsOp()
        ds_name = cfg.TRAIN.DATASETS[0] if len(cfg.TRAIN.DATASETS) else cfg.TEST.DATASETS[0]
    
    def get_roi_inds(self, det_labels, lbls):
        lbl_set = np.array(lbls)
        inds = np.where(np.isin(det_labels, lbl_set))[0]
        return inds

    def remove_self_pairs(self, det_size, sbj_inds, obj_inds):
        mask = np.ones(sbj_inds.shape[0], dtype=bool)
        for i in range(det_size):
            mask[i + det_size * i] = False
        keeps = np.where(mask)[0]
        sbj_inds = sbj_inds[keeps]
        obj_inds = obj_inds[keeps]
        return sbj_inds, obj_inds

    def forward(self, det_rois, det_labels, det_scores, im_info, dataset_name, roidb=None):
        """
        det_rois: feature maps from the backbone network. (Variable)
        im_info: (CPU Variable)
        roidb: (list of ndarray)
        """
        
        # Get pairwise proposals first
        if roidb is not None:
            # we always feed one image per batch during training
            assert len(roidb) == 1

        sbj_inds = np.repeat(np.arange(det_rois.shape[0]), det_rois.shape[0])
        obj_inds = np.tile(np.arange(det_rois.shape[0]), det_rois.shape[0])
        # remove self paired rois
        if det_rois.shape[0] > 1:  # no pairs to remove when there is at most one detection
            sbj_inds, obj_inds = self.remove_self_pairs(det_rois.shape[0], sbj_inds, obj_inds)
        sbj_rois = det_rois[sbj_inds]
        obj_rois = det_rois[obj_inds]
            
        im_scale = im_info.data.numpy()[:, 2][0]
        sbj_boxes = sbj_rois[:, 1:] / im_scale
        obj_boxes = obj_rois[:, 1:] / im_scale
        # filters out those roi pairs whose boxes are not overlapping in the original scales
        if cfg.MODEL.USE_OVLP_FILTER:
            ovlp_so = box_utils_rel.bbox_pair_overlaps(
                sbj_boxes.astype(dtype=np.float32, copy=False),
                obj_boxes.astype(dtype=np.float32, copy=False))
            ovlp_inds = np.where(ovlp_so > 0)[0]
            sbj_inds = sbj_inds[ovlp_inds]
            obj_inds = obj_inds[ovlp_inds]
            sbj_rois = sbj_rois[ovlp_inds]
            obj_rois = obj_rois[ovlp_inds]
            sbj_boxes = sbj_boxes[ovlp_inds]
            obj_boxes = obj_boxes[ovlp_inds]
            
        return_dict = {}
        if self.training:
            # Add binary relationships
            blobs_out = self.RelPN_GenerateProposalLabels(sbj_rois, obj_rois, det_rois, roidb, im_info)
            return_dict.update(blobs_out)
        else:
            sbj_labels = det_labels[sbj_inds]
            obj_labels = det_labels[obj_inds]
            sbj_scores = det_scores[sbj_inds]
            obj_scores = det_scores[obj_inds]
            rel_rois = box_utils_rel.rois_union(sbj_rois, obj_rois)
            return_dict['det_rois'] = det_rois
            return_dict['sbj_inds'] = sbj_inds
            return_dict['obj_inds'] = obj_inds
            return_dict['sbj_rois'] = sbj_rois
            return_dict['obj_rois'] = obj_rois
            return_dict['rel_rois'] = rel_rois
            return_dict['sbj_labels'] = sbj_labels
            return_dict['obj_labels'] = obj_labels
            return_dict['sbj_scores'] = sbj_scores
            return_dict['obj_scores'] = obj_scores
            return_dict['fg_size'] = np.array([sbj_rois.shape[0]], dtype=np.int32)

            im_scale = im_info.data.numpy()[:, 2][0]
            im_w = im_info.data.numpy()[:, 1][0]
            im_h = im_info.data.numpy()[:, 0][0]
            if cfg.MODEL.USE_SPATIAL_FEAT:
                spt_feat = box_utils_rel.get_spt_features(sbj_boxes, obj_boxes, im_w, im_h)
                return_dict['spt_feat'] = spt_feat
            if cfg.MODEL.USE_FREQ_BIAS or cfg.MODEL.RUN_BASELINE:
                return_dict['all_sbj_labels_int32'] = sbj_labels.astype(np.int32, copy=False) - 1  # det_labels start from 1
                return_dict['all_obj_labels_int32'] = obj_labels.astype(np.int32, copy=False) - 1  # det_labels start from 1
            if cfg.FPN.FPN_ON and cfg.FPN.MULTILEVEL_ROIS:
                lvl_min = cfg.FPN.ROI_MIN_LEVEL
                lvl_max = cfg.FPN.ROI_MAX_LEVEL
                # when use min_rel_area, the same sbj/obj area could be mapped to different feature levels
                # when they are associated with different relationships
                # Thus we cannot get det_rois features then gather sbj/obj features
                # The only way is gather sbj/obj per relationship, thus need to return sbj_rois/obj_rois
                rois_blob_names = ['det_rois', 'rel_rois']
                for rois_blob_name in rois_blob_names:
                    # Add per FPN level roi blobs named like: <rois_blob_name>_fpn<lvl>
                    target_lvls = fpn_utils.map_rois_to_fpn_levels(
                        return_dict[rois_blob_name][:, 1:5], lvl_min, lvl_max)
                    fpn_utils.add_multilevel_roi_blobs(
                        return_dict, rois_blob_name, return_dict[rois_blob_name], target_lvls,
                        lvl_min, lvl_max)

        return return_dict









