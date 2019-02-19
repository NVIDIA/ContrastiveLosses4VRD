# Written by Ji Zhang in 2019

import os
import numpy as np
import logging
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F

from core.config import cfg
import nn as mynn
import torchvision.models as models

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------- #
# VGG16 architecture
# ---------------------------------------------------------------------------- #

vgg = models.vgg16()
if cfg.VGG16.IMAGENET_PRETRAINED_WEIGHTS != '':
    logger.info("Loading imagenet pretrained weights from %s", cfg.VGG16.IMAGENET_PRETRAINED_WEIGHTS)
    state_dict = torch.load(cfg.VGG16.IMAGENET_PRETRAINED_WEIGHTS)
    vgg.load_state_dict({k:v for k, v in state_dict.items() if k in vgg.state_dict()})

class VGG16_conv_body(nn.Module):
    def __init__(self):
        super().__init__()
        self.num_layers = 16
        self.spatial_scale = 1. / 16.  # final feature scale wrt. original image scale
        self.dim_out = 512

        self._init_modules()

    def _init_modules(self):
        
        # not using the last maxpool layer
        self.convs = nn.Sequential(*list(vgg.features._modules.values())[:-1])
        
        for layer in range(10):
            for p in self.convs[layer].parameters(): p.requires_grad = False

    def forward(self, x):
        
        return self.convs(x)


class VGG16_roi_conv5_head(nn.Module):
    def __init__(self, dim_in, roi_xform_func, spatial_scale):
        super().__init__()
        self.roi_xform = roi_xform_func
        self.spatial_scale = spatial_scale

        self.dim_out = 4096
        self.dim_roi_out = dim_in  # 512

        self._init_modules()

    def _init_modules(self):

        self.heads = nn.Sequential(*list(vgg.classifier._modules.values())[:-1])

    def forward(self, x, rpn_ret, rois_name='rois', use_relu=True):
        x = self.roi_xform(
            x, rpn_ret,
            blob_rois=rois_name,
            method=cfg.FAST_RCNN.ROI_XFORM_METHOD,
            resolution=7,
            spatial_scale=self.spatial_scale,
            sampling_ratio=cfg.FAST_RCNN.ROI_XFORM_SAMPLING_RATIO
        )

        feat = x.view(x.size(0), -1)
        
        if use_relu:
            for layer in list(self.heads.children()):
                feat = layer(feat)
        else:
            # not use the last Drop-out and ReLU in fc7 (keep it the same with Rawan's paper)
            for layer in list(self.heads.children())[:-2]:
                feat = layer(feat)
        
        return feat
