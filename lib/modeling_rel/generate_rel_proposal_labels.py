# Adapted from Detectron.pytorch/lib/modeling/generate_proposal_labels.py
# for this project by Ji Zhang, 2019

from torch import nn

from core.config import cfg
from datasets_rel import json_dataset_rel
from roi_data_rel.fast_rcnn_rel import add_rel_blobs


class GenerateRelProposalLabelsOp(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, sbj_rois, obj_rois, det_rois, roidb, im_info):
        
        im_scales = im_info.data.numpy()[:, 2]
        # For historical consistency with the original Faster R-CNN
        # implementation we are *not* filtering crowd proposals.
        # This choice should be investigated in the future (it likely does
        # not matter).
        # Note: crowd_thresh=0 will ignore _filter_crowd_proposals
        json_dataset_rel.add_rel_proposals(roidb, sbj_rois, obj_rois, det_rois, im_scales)
        output_blob_names = ['sbj_rois', 'obj_rois', 'rel_rois', 'fg_prd_labels_int32', 'all_prd_labels_int32', 'fg_size']
        if cfg.MODEL.USE_SPATIAL_FEAT:
            output_blob_names += ['spt_feat']
        if cfg.MODEL.USE_FREQ_BIAS:
            output_blob_names += ['all_sbj_labels_int32']
            output_blob_names += ['all_obj_labels_int32']
        if cfg.MODEL.USE_NODE_CONTRASTIVE_LOSS or cfg.MODEL.USE_NODE_CONTRASTIVE_SO_AWARE_LOSS or cfg.MODEL.USE_NODE_CONTRASTIVE_P_AWARE_LOSS:
            output_blob_names += ['binary_labels_sbj_pos_int32',
                                  'sbj_rois_sbj_pos', 'obj_rois_sbj_pos', 'rel_rois_sbj_pos',
                                  'spt_feat_sbj_pos',
                                  'sbj_labels_sbj_pos_int32', 'obj_labels_sbj_pos_int32', 'prd_labels_sbj_pos_int32',
                                  'sbj_labels_sbj_pos_fg_int32', 'obj_labels_sbj_pos_fg_int32',
                                  'inds_unique_sbj_pos',
                                  'inds_reverse_sbj_pos',
                                  'binary_labels_obj_pos_int32',
                                  'sbj_rois_obj_pos', 'obj_rois_obj_pos', 'rel_rois_obj_pos',
                                  'spt_feat_obj_pos',
                                  'sbj_labels_obj_pos_int32', 'obj_labels_obj_pos_int32', 'prd_labels_obj_pos_int32',
                                  'sbj_labels_obj_pos_fg_int32', 'obj_labels_obj_pos_fg_int32',
                                  'inds_unique_obj_pos',
                                  'inds_reverse_obj_pos']
        blobs = {k: [] for k in output_blob_names}
        
        add_rel_blobs(blobs, im_scales, roidb)

        return blobs
