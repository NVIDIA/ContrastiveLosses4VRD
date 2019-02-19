# Adapted from Detectron.pytorch/lib/modeling/generate_proposal_labels.py
# for this project by Ji Zhang, 2019

from torch import nn

from core.config import cfg
from datasets_rel import json_dataset_att
from roi_data_rel.fast_rcnn_att import add_att_blobs


class GenerateAttProposalLabelsOp(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, det_rois, roidb, im_info):
        
        im_scales = im_info.data.numpy()[:, 2]
        # For historical consistency with the original Faster R-CNN
        # implementation we are *not* filtering crowd proposals.
        # This choice should be investigated in the future (it likely does
        # not matter).
        # Note: crowd_thresh=0 will ignore _filter_crowd_proposals
        json_dataset_att.add_att_proposals(roidb, det_rois, im_scales)
        output_blob_names = ['obj_rois', 'fg_att_labels_int32', 'all_att_labels_int32', 'fg_size']
        if cfg.MODEL.USE_FREQ_BIAS:
            output_blob_names += ['all_obj_labels_int32']
        blobs = {k: [] for k in output_blob_names}
        
        add_att_blobs(blobs, im_scales, roidb)
        
        return blobs
