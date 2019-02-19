# Written by Ji Zhang in 2019

import collections
import numpy as np
import logging

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init

from core.config import cfg
import utils.net as net_utils
import modeling.ResNet as ResNet
from modeling.generate_anchors import generate_anchors
from modeling.generate_proposals import GenerateProposalsOp
from modeling.collect_and_distribute_fpn_rpn_proposals import CollectAndDistributeFpnRpnProposalsOp
import nn as mynn

logger = logging.getLogger(__name__)


class rel_pyramid_module(nn.Module):
    def __init__(self, num_backbone_stages):
        super().__init__()
        
        fpn_dim = cfg.FPN.DIM
        self.num_backbone_stages = num_backbone_stages
        
        self.prd_conv_lateral = nn.ModuleList()
        for i in range(self.num_backbone_stages):
            if cfg.FPN.USE_GN:
                self.prd_conv_lateral.append(nn.Sequential(
                    nn.Conv2d(fpn_dim, fpn_dim, 1, 1, 0, bias=False),
                    nn.GroupNorm(net_utils.get_group_gn(fpn_dim), fpn_dim,
                                 eps=cfg.GROUP_NORM.EPSILON)))
            else:
                self.prd_conv_lateral.append(nn.Conv2d(fpn_dim, fpn_dim, 1, 1, 0))
        
        self.posthoc_modules = nn.ModuleList()
        for i in range(self.num_backbone_stages):
            if cfg.FPN.USE_GN:
                self.posthoc_modules.append(nn.Sequential(
                    nn.Conv2d(fpn_dim, fpn_dim, 3, 1, 1, bias=False),
                    nn.GroupNorm(net_utils.get_group_gn(fpn_dim), fpn_dim,
                                 eps=cfg.GROUP_NORM.EPSILON)))
            else:
                self.posthoc_modules.append(nn.Conv2d(fpn_dim, fpn_dim, 3, 1, 1))
        
        self._init_weights()
        
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                mynn.init.XavierFill(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, blob_conv):
        # blob_conv is in the order (P5, P4, P3, P2)
        rel_lateral_inner_blob = None
        rel_lateral_output_blobs = []
        for i in range(self.num_backbone_stages):
            if rel_lateral_inner_blob is not None:
                bu = F.max_pool2d(rel_lateral_inner_blob, 2, stride=2)
                rel_lateral_inner_blob = \
                    self.prd_conv_lateral[i](blob_conv[-1 - i]) + bu
            else:
                rel_lateral_inner_blob = \
                    self.prd_conv_lateral[i](blob_conv[-1 - i])
            rel_lateral_output_blobs.append(self.posthoc_modules[i](rel_lateral_inner_blob))
        
        # the output is in the order of (P2, P3, P4, P5), we need to recover it back to (P5, P4, P3, P2)
        rel_lateral_output_blobs.reverse()
        return rel_lateral_output_blobs
