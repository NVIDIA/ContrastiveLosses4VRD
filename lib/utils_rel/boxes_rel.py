# Adapted by Ji Zhang in 2019 for this project
# Based on Detectron.pytorch/lib/utils/boxes.py
#
# Original license text below:
# 
#############################################################################
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
# Fast/er R-CNN
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# --------------------------------------------------------

"""Box manipulation functions. The internal Detectron box format is
[x1, y1, x2, y2] where (x1, y1) specify the top-left box corner and (x2, y2)
specify the bottom-right box corner. Boxes from external sources, e.g.,
datasets, may be in other formats (such as [x, y, w, h]) and require conversion.

This module uses a convention that may seem strange at first: the width of a box
is computed as x2 - x1 + 1 (likewise for height). The "+ 1" dates back to old
object detection days when the coordinates were integer pixel indices, rather
than floating point coordinates in a subpixel coordinate frame. A box with x2 =
x1 and y2 = y1 was taken to include a single pixel, having a width of 1, and
hence requiring the "+ 1". Now, most datasets will likely provide boxes with
floating point coordinates and the width should be more reasonably computed as
x2 - x1.

In practice, as long as a model is trained and tested with a consistent
convention either decision seems to be ok (at least in our experience on COCO).
Since we have a long history of training models with the "+ 1" convention, we
are reluctant to change it even if our modern tastes prefer not to use it.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import warnings
import numpy as np

from core.config import cfg
import utils_rel.cython_bbox_rel as cython_bbox_rel
from utils.boxes import bbox_transform_inv


bbox_pair_overlaps = cython_bbox_rel.bbox_pair_overlaps


def get_spt_features(boxes1, boxes2, width, height):
    boxes_u = boxes_union(boxes1, boxes2)
    spt_feat_1 = get_box_feature(boxes1, width, height)
    spt_feat_2 = get_box_feature(boxes2, width, height)
    spt_feat_12 = get_pair_feature(boxes1, boxes2)
    spt_feat_1u = get_pair_feature(boxes1, boxes_u)
    spt_feat_u2 = get_pair_feature(boxes_u, boxes2)
    return np.hstack((spt_feat_12, spt_feat_1u, spt_feat_u2, spt_feat_1, spt_feat_2))

    
def get_pair_feature(boxes1, boxes2):
    delta_1 = bbox_transform_inv(boxes1, boxes2)
    delta_2 = bbox_transform_inv(boxes2, boxes1)
    spt_feat = np.hstack((delta_1, delta_2[:, :2]))
    return spt_feat


def get_box_feature(boxes, width, height):
    f1 = boxes[:, 0] / width
    f2 = boxes[:, 1] / height
    f3 = boxes[:, 2] / width
    f4 = boxes[:, 3] / height
    f5 = (boxes[:, 2] - boxes[:, 0] + 1) * (boxes[:, 3] - boxes[:, 1] + 1) / (width * height)
    return np.vstack((f1, f2, f3, f4, f5)).transpose()
    

def boxes_union(boxes1, boxes2):
    assert boxes1.shape == boxes2.shape
    xmin = np.minimum(boxes1[:, 0], boxes2[:, 0])
    ymin = np.minimum(boxes1[:, 1], boxes2[:, 1])
    xmax = np.maximum(boxes1[:, 2], boxes2[:, 2])
    ymax = np.maximum(boxes1[:, 3], boxes2[:, 3])
    return np.vstack((xmin, ymin, xmax, ymax)).transpose()


def rois_union(rois1, rois2):
    assert (rois1[:, 0] == rois2[:, 0]).all()
    xmin = np.minimum(rois1[:, 1], rois2[:, 1])
    ymin = np.minimum(rois1[:, 2], rois2[:, 2])
    xmax = np.maximum(rois1[:, 3], rois2[:, 3])
    ymax = np.maximum(rois1[:, 4], rois2[:, 4])
    return np.vstack((rois1[:, 0], xmin, ymin, xmax, ymax)).transpose()


def boxes_intersect(boxes1, boxes2):
    assert boxes1.shape == boxes2.shape
    xmin = np.maximum(boxes1[:, 0], boxes2[:, 0])
    ymin = np.maximum(boxes1[:, 1], boxes2[:, 1])
    xmax = np.minimum(boxes1[:, 2], boxes2[:, 2])
    ymax = np.minimum(boxes1[:, 3], boxes2[:, 3])
    return np.vstack((xmin, ymin, xmax, ymax)).transpose()


def rois_intersect(rois1, rois2):
    assert (rois1[:, 0] == rois2[:, 0]).all()
    xmin = np.maximum(rois1[:, 1], rois2[:, 1])
    ymin = np.maximum(rois1[:, 2], rois2[:, 2])
    xmax = np.minimum(rois1[:, 3], rois2[:, 3])
    ymax = np.minimum(rois1[:, 4], rois2[:, 4])
    return np.vstack((rois1[:, 0], xmin, ymin, xmax, ymax)).transpose()


def y1y2x1x2_to_x1y1x2y2(y1y2x1x2):
    x1 = y1y2x1x2[2]
    y1 = y1y2x1x2[0]
    x2 = y1y2x1x2[3]
    y2 = y1y2x1x2[1]
    return [x1, y1, x2, y2]
