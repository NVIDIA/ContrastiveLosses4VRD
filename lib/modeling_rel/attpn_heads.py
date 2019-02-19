# Written by Ji Zhang in 2019

import numpy as np
from numpy import linalg as la
import json
import logging

from torch import nn
from torch.nn import init
import torch.nn.functional as F

from core.config import cfg
from modeling_rel.generate_att_proposal_labels import GenerateAttProposalLabelsOp
import modeling.FPN as FPN
import utils.fpn as fpn_utils


logger = logging.getLogger(__name__)


def generic_attpn_outputs():
    return single_scale_attpn_outputs()


class single_scale_attpn_outputs(nn.Module):
    """Add AttPN outputs to a single scale model (i.e., no FPN)."""
    def __init__(self, word2vec_model=None):
        super().__init__()
        
        self.AttPN_GenerateProposalLabels = GenerateAttProposalLabelsOp()

        ds_name = cfg.TRAIN.DATASETS[0] if len(cfg.TRAIN.DATASETS) else cfg.TEST.DATASETS[0]
        assert ds_name.find('oi') >= 0
        with open(cfg.DATA_DIR + '/openimages_v4/rel/all_23_att_object_labels.json') as f:
            self.att_obj_lbls = json.load(f)
    
    def get_roi_inds(self, det_labels, lbls):
        lbl_set = np.array(lbls)
        inds = np.where(np.isin(det_labels, lbl_set))[0]
        return inds

    def forward(self, det_rois, det_labels, det_scores, im_info, dataset_name, roidb=None):
        """
        det_rois: feature maps from the backbone network. (Variable)
        im_info: (CPU Variable)
        roidb: (list of ndarray)
        """

        return_dict = {}
        if self.training:
            # Add attributes
            att_blobs_out = self.AttPN_GenerateProposalLabels(det_rois, roidb, im_info)
            return_dict.update(att_blobs_out)
        else:
            att_inds = self.get_roi_inds(det_labels - 1, self.att_obj_lbls)  # det_labels start from 1
            att_det_rois = det_rois[att_inds]
            att_det_labels = det_labels[att_inds]
            att_det_scores = det_scores[att_inds]
            return_dict['obj_rois'] = att_det_rois
            return_dict['obj_labels'] = att_det_labels  # att_det_labels start from 1
            return_dict['obj_scores'] = att_det_scores
            return_dict['all_obj_labels_int32'] = att_det_labels.astype(np.int32, copy=False) - 1  # att_det_labels start from 1
            return_dict['fg_size'] = np.array([att_det_rois.shape[0]], dtype=np.int32)
            if cfg.FPN.FPN_ON and cfg.FPN.MULTILEVEL_ROIS:
                # Get target level for each roi
                # Recall blob rois are in (batch_idx, x1, y1, x2, y2) format, hence take
                # the box coordinates from columns 1:5
                rois_blob_name = 'obj_rois'
                target_lvls = fpn_utils.map_rois_to_fpn_levels(
                    return_dict[rois_blob_name][:, 1:5], cfg.FPN.ROI_MIN_LEVEL, cfg.FPN.ROI_MAX_LEVEL)
                # Add per FPN level roi blobs named like: <rois_blob_name>_fpn<lvl>
                fpn_utils.add_multilevel_roi_blobs(
                    return_dict, rois_blob_name, return_dict[rois_blob_name], target_lvls,
                    cfg.FPN.ROI_MIN_LEVEL, cfg.FPN.ROI_MAX_LEVEL)

        return return_dict
