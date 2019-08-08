# Some functions are adapted from Rowan Zellers:
# https://github.com/rowanz/neural-motifs
# Get counts of all of the examples in the dataset. Used for creating the baseline
# dictionary model

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import numpy as np
import json

import utils.boxes as box_utils
import utils_rel.boxes_rel as box_utils_rel
from core.config import cfg

from datasets_rel.dataset_catalog_rel import ANN_FN2
from datasets_rel.dataset_catalog_rel import DATASETS


# This function is adapted from Rowan Zellers:
# https://github.com/rowanz/neural-motifs/blob/master/lib/get_dataset_counts.py
# Modified for this project
def get_rel_counts(ds_name, must_overlap=True):
    """
    Get counts of all of the relations. Used for modeling directly P(rel | o1, o2)
    :param train_data: 
    :param must_overlap: 
    :return: 
    """

#     if ds_name.find('vg') >= 0:
#         with open(DATASETS['vg_train'][ANN_FN2]) as f:
#             train_data = json.load(f)
#     elif ds_name.find('oi') >= 0:
#         with open(DATASETS['oi_rel_train'][ANN_FN2]) as f:
#             train_data = json.load(f)
#     elif ds_name.find('vrd') >= 0:
#         with open(DATASETS['vrd_train'][ANN_FN2]) as f:
#             train_data = json.load(f)
#     elif ds_name.find('gqa') >= 0:
#         with open(DATASETS['gqa_train'][ANN_FN2]) as f:
#             train_data = json.load(f)
#     else:
#         raise NotImplementedError

    if len(cfg.TRAIN.DATASETS) > 0:
        with open(DATASETS[cfg.TRAIN.DATASETS[0]][ANN_FN2]) as f:
            train_data = json.load(f)
    else:
        ds_keywords = cfg.TEST.DATASETS[0].split('_')
        if ds_keywords[-1] == 'val' or ds_keywords[-1] == 'test' or ds_keywords[-1] == 'all':
            ds_keywords = ds_keywords[:-1]
        elif ds_keywords[-2] == 'of':  # those "x_of_3" test json files
            ds_keywords = ds_keywords[:-3]
        ds_name = '_'.join(ds_keywords + ['train'])
        with open(DATASETS[ds_name][ANN_FN2]) as f:
            train_data = json.load(f)

    fg_matrix = np.zeros((
        cfg.MODEL.NUM_CLASSES - 1,  # not include background
        cfg.MODEL.NUM_CLASSES - 1,  # not include background
        cfg.MODEL.NUM_PRD_CLASSES + 1,  # include background
    ), dtype=np.int64)

    bg_matrix = np.zeros((
        cfg.MODEL.NUM_CLASSES - 1,  # not include background
        cfg.MODEL.NUM_CLASSES - 1,  # not include background
    ), dtype=np.int64)
    
    for _, im_rels in train_data.items():
        # get all object boxes
        gt_box_to_label = {}
        for i, rel in enumerate(im_rels):
            sbj_box = box_utils_rel.y1y2x1x2_to_x1y1x2y2(rel['subject']['bbox'])
            obj_box = box_utils_rel.y1y2x1x2_to_x1y1x2y2(rel['object']['bbox'])
            sbj_lbl = rel['subject']['category']  # not include background
            obj_lbl = rel['object']['category']  # not include background
            prd_lbl = rel['predicate']  # not include background
            if tuple(sbj_box) not in gt_box_to_label:
                gt_box_to_label[tuple(sbj_box)] = sbj_lbl
            if tuple(obj_box) not in gt_box_to_label:
                gt_box_to_label[tuple(obj_box)] = obj_lbl
            
            fg_matrix[sbj_lbl, obj_lbl, prd_lbl + 1] += 1
        
        if cfg.MODEL.USE_OVLP_FILTER:
            if len(gt_box_to_label):
                gt_boxes = np.array(list(gt_box_to_label.keys()), dtype=np.int32)
                gt_classes = np.array(list(gt_box_to_label.values()), dtype=np.int32)
                o1o2_total = gt_classes[np.array(
                    box_filter(gt_boxes, must_overlap=must_overlap), dtype=int)]
                for (o1, o2) in o1o2_total:
                    bg_matrix[o1, o2] += 1
        else:
            # consider all pairs of boxes, overlapped or non-overlapped
            for b1, l1 in gt_box_to_label.items():
                for b2, l2 in gt_box_to_label.items():
                    if b1 == b2:
                        continue
                    bg_matrix[l1, l2] += 1

    return fg_matrix, bg_matrix


# This function is adapted from Rowan Zellers:
# https://github.com/rowanz/neural-motifs/blob/master/lib/get_dataset_counts.py
# Modified for this project
def box_filter(boxes, must_overlap=False):
    """ Only include boxes that overlap as possible relations. 
    If no overlapping boxes, use all of them."""
    n_cands = boxes.shape[0]

    overlaps = box_utils.bbox_overlaps(boxes.astype(np.float32), boxes.astype(np.float32)) > 0
    np.fill_diagonal(overlaps, 0)

    all_possib = np.ones_like(overlaps, dtype=np.bool)
    np.fill_diagonal(all_possib, 0)

    if must_overlap:
        possible_boxes = np.column_stack(np.where(overlaps))

        if possible_boxes.size == 0:
            possible_boxes = np.column_stack(np.where(all_possib))
    else:
        possible_boxes = np.column_stack(np.where(all_possib))
    return possible_boxes
