# Written by Ji Zhang in 2019

import numpy as np
from numpy import linalg as la
import math
import logging
import json

import torch
from torch import nn
from torch.nn import init
import torch.nn.functional as F
from torch.autograd import Variable
import nn as mynn

from core.config import cfg
from modeling_rel.sparse_targets_att import FrequencyBias

logger = logging.getLogger(__name__)


class attdn_head(nn.Module):
    """Add AttDN outputs to a single scale model (i.e., no FPN)."""
    def __init__(self, dim_in):  # dim_in is 4096 for VGG16 and 1024 for FPN
        super().__init__()
        self.dim_in = dim_in
        if cfg.MODEL.USE_BG:
            num_att_classes = cfg.MODEL.NUM_ATT_CLASSES + 1
        else:
            num_att_classes = cfg.MODEL.NUM_ATT_CLASSES
            
        if cfg.MODEL.RUN_BASELINE:
            # only run it on testing mode
            self.freq_bias = FrequencyBias()
            return
        
        self.att_cls_scores = nn.Linear(dim_in, num_att_classes)  # 5 foreground attributes and 1 background
        
        if cfg.MODEL.USE_FREQ_BIAS:
            # Assume we are training/testing on only one dataset
            if len(cfg.TRAIN.DATASETS):
                self.freq_bias = FrequencyBias()
            else:
                self.freq_bias = FrequencyBias()
        
        self._init_weights()
        
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                # nn.init.kaiming_normal_(m.weight, mode='fan_out')
                mynn.init.XavierFill(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    # x will be the fc7 feature of att_rois
    def forward(self, x, obj_labels=None, fg_size=None):

        device_id = x.get_device()
        if obj_labels is not None:
            obj_labels = Variable(torch.from_numpy(obj_labels.astype('int64'))).cuda(device_id)
        if fg_size is not None:
            fg_size = Variable(torch.from_numpy(fg_size.astype('int64'))).cuda(device_id)

        if cfg.MODEL.RUN_BASELINE:
            assert obj_labels is not None
            att_cls_scores = self.freq_bias.att_index_with_labels(obj_labels)
            att_cls_scores = F.softmax(att_cls_scores, dim=1)
            return att_cls_scores
        
        if x.dim() == 4:
            x = x.squeeze(3).squeeze(2)

        att_cls_scores = self.att_cls_scores(x)
        if cfg.MODEL.USE_FREQ_BIAS:
            assert obj_labels is not None
            att_cls_scores = att_cls_scores + self.freq_bias.att_index_with_labels(obj_labels)
            
        if not self.training:
            att_cls_scores = F.softmax(att_cls_scores, dim=1)
        
        return att_cls_scores

    
def attdn_losses(att_cls_scores, att_labels_int32):
    device_id = att_cls_scores.get_device()
    att_labels = Variable(torch.from_numpy(att_labels_int32.astype('int64'))).cuda(device_id)
    loss_cls_att = F.cross_entropy(att_cls_scores, att_labels)

    # class accuracy
    att_cls_preds = att_cls_scores.max(dim=1)[1].type_as(att_labels)
    accuracy_cls_att = att_cls_preds.eq(att_labels).float().mean(dim=0)

    return loss_cls_att, accuracy_cls_att


def attdn_o_losses(obj_cls_scores, obj_labels_int32):
    device_id = obj_cls_scores.get_device()
    
    obj_labels = Variable(torch.from_numpy(obj_labels_int32.astype('int64'))).cuda(device_id)
    loss_cls_obj = F.cross_entropy(obj_cls_scores, obj_labels)
    obj_cls_preds = obj_cls_scores.max(dim=1)[1].type_as(obj_labels)
    accuracy_cls_obj = obj_cls_preds.eq(obj_labels).float().mean(dim=0)
    
    return loss_cls_obj, accuracy_cls_obj
