# Adapted from Detectron.pytorch/lib/datasets/json_dataset.py
# for this project by Ji Zhang, 2019
#-----------------------------------------------------------------------------
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

"""Representation of the standard COCO json dataset format.

When working with a new dataset, we strongly suggest to convert the dataset into
the COCO json format and use the existing code; it is not recommended to write
code to support new dataset formats.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import copy
from six.moves import cPickle as pickle
import logging
import numpy as np
import os
import scipy.sparse
import json

# Must happen before importing COCO API (which imports matplotlib)
import utils.env as envu
envu.set_up_matplotlib()
# COCO API
from pycocotools import mask as COCOmask
from pycocotools.coco import COCO

import utils.boxes as box_utils
import utils_rel.boxes_rel as box_utils_rel
from core.config import cfg
from utils.timer import Timer
from .dataset_catalog_rel import ANN_FN
from .dataset_catalog_rel import ANN_FN2
from .dataset_catalog_rel import ANN_FN3
from .dataset_catalog_rel import DATASETS
from .dataset_catalog_rel import IM_DIR
from .dataset_catalog_rel import IM_PREFIX

logger = logging.getLogger(__name__)


class JsonDatasetRel(object):
    """A class representing a COCO json dataset."""

    def __init__(self, name):
        assert name in DATASETS.keys(), \
            'Unknown dataset name: {}'.format(name)
        assert os.path.exists(DATASETS[name][IM_DIR]), \
            'Image directory \'{}\' not found'.format(DATASETS[name][IM_DIR])
        assert os.path.exists(DATASETS[name][ANN_FN]), \
            'Annotation file \'{}\' not found'.format(DATASETS[name][ANN_FN])
        logger.debug('Creating: {}'.format(name))
        self.name = name
        self.image_directory = DATASETS[name][IM_DIR]
        self.image_prefix = (
            '' if IM_PREFIX not in DATASETS[name] else DATASETS[name][IM_PREFIX]
        )
        self.COCO = COCO(DATASETS[name][ANN_FN])
        self.debug_timer = Timer()
        # Set up dataset classes
        category_ids = self.COCO.getCatIds()
        categories = [c['name'] for c in self.COCO.loadCats(category_ids)]
        self.category_to_id_map = dict(zip(categories, category_ids))
        self.classes = ['__background__'] + categories
        self.num_classes = len(self.classes)
        self.json_category_id_to_contiguous_id = {
            v: i + 1
            for i, v in enumerate(self.COCO.getCatIds())
        }
        self.contiguous_category_id_to_json_id = {
            v: k
            for k, v in self.json_category_id_to_contiguous_id.items()
        }
        self._init_keypoints()

        assert ANN_FN2 in DATASETS[name] and ANN_FN3 in DATASETS[name]
        with open(DATASETS[name][ANN_FN2]) as f:
            self.rel_anns = json.load(f)
        with open(DATASETS[name][ANN_FN3]) as f:
            prd_categories = json.load(f)
        self.obj_classes = self.classes[1:]  # excludes background for now
        self.num_obj_classes = len(self.obj_classes)
        # self.prd_classes = ['__background__'] + prd_categories
        self.prd_classes = prd_categories  # excludes background for now
        self.num_prd_classes = len(self.prd_classes)

    @property
    def cache_path(self):
        cache_path = os.path.abspath(os.path.join(cfg.DATA_DIR, 'cache'))
        if not os.path.exists(cache_path):
            os.makedirs(cache_path)
        return cache_path

    @property
    def valid_cached_keys(self):
        """ Can load following key-ed values from the cached roidb file

        'image'(image path) and 'flipped' values are already filled on _prep_roidb_entry,
        so we don't need to overwrite it again.
        """
        keys = ['dataset_name',
                'boxes', 'segms', 'gt_classes', 'seg_areas', 'gt_overlaps',
                'is_crowd', 'box_to_gt_ind_map',
                'sbj_gt_boxes', 'sbj_gt_classes', 'obj_gt_boxes', 'obj_gt_classes', 'prd_gt_classes',
                'sbj_gt_overlaps', 'obj_gt_overlaps', 'prd_gt_overlaps', 'pair_to_gt_ind_map']
        if self.keypoints is not None:
            keys += ['gt_keypoints', 'has_visible_keypoints']
        return keys

    def get_roidb(
            self,
            gt=False,
            proposal_file=None,
            min_proposal_size=2,
            proposal_limit=-1,
            crowd_filter_thresh=0
        ):
        """Return an roidb corresponding to the json dataset. Optionally:
           - include ground truth boxes in the roidb
           - add proposals specified in a proposals file
           - filter proposals based on a minimum side length
           - filter proposals that intersect with crowd regions
        """
        assert gt is True or crowd_filter_thresh == 0, \
            'Crowd filter threshold must be 0 if ground-truth annotations ' \
            'are not included.'
        image_ids = self.COCO.getImgIds()
        image_ids.sort()
        if cfg.DEBUG:
            roidb = copy.deepcopy(self.COCO.loadImgs(image_ids))[:100]
        else:
            roidb = copy.deepcopy(self.COCO.loadImgs(image_ids))
        new_roidb = []
        for entry in roidb:
            # In OpenImages_v4, the detection-annotated images are more than relationship
            # annotated images, hence the need to check
            if entry['file_name'] in self.rel_anns:
                self._prep_roidb_entry(entry)
                new_roidb.append(entry)
        roidb = new_roidb
        if gt:
            # Include ground-truth object annotations
            cache_filepath = os.path.join(self.cache_path, self.name + '_rel_gt_roidb.pkl')
            if os.path.exists(cache_filepath) and not cfg.DEBUG:
                self.debug_timer.tic()
                self._add_gt_from_cache(roidb, cache_filepath)
                logger.debug(
                    '_add_gt_from_cache took {:.3f}s'.
                    format(self.debug_timer.toc(average=False))
                )
            else:
                self.debug_timer.tic()
                for entry in roidb:
                    self._add_gt_annotations(entry)    
                logger.debug(
                    '_add_gt_annotations took {:.3f}s'.
                    format(self.debug_timer.toc(average=False))
                )
                if not cfg.DEBUG:
                    with open(cache_filepath, 'wb') as fp:
                        pickle.dump(roidb, fp, pickle.HIGHEST_PROTOCOL)
                    logger.info('Cache ground truth roidb to %s', cache_filepath)
        if proposal_file is not None:
            # Include proposals from a file
            self.debug_timer.tic()
            self._add_proposals_from_file(
                roidb, proposal_file, min_proposal_size, proposal_limit,
                crowd_filter_thresh
            )
            logger.debug(
                '_add_proposals_from_file took {:.3f}s'.
                format(self.debug_timer.toc(average=False))
            )
        _add_class_assignments(roidb)
        return roidb

    def _prep_roidb_entry(self, entry):
        """Adds empty metadata fields to an roidb entry."""
        # Reference back to the parent dataset
        entry['dataset'] = self
        # Make file_name an abs path
        im_path = os.path.join(
            self.image_directory, self.image_prefix + entry['file_name']
        )
        assert os.path.exists(im_path), 'Image \'{}\' not found'.format(im_path)
        entry['image'] = im_path
        entry['flipped'] = False
        entry['has_visible_keypoints'] = False
        # Empty placeholders
        entry['boxes'] = np.empty((0, 4), dtype=np.float32)
        entry['segms'] = []
        entry['gt_classes'] = np.empty((0), dtype=np.int32)
        entry['seg_areas'] = np.empty((0), dtype=np.float32)
        entry['gt_overlaps'] = scipy.sparse.csr_matrix(
            np.empty((0, self.num_classes), dtype=np.float32)
        )
        entry['is_crowd'] = np.empty((0), dtype=np.bool)
        # 'box_to_gt_ind_map': Shape is (#rois). Maps from each roi to the index
        # in the list of rois that satisfy np.where(entry['gt_classes'] > 0)
        entry['box_to_gt_ind_map'] = np.empty((0), dtype=np.int32)
        if self.keypoints is not None:
            entry['gt_keypoints'] = np.empty(
                (0, 3, self.num_keypoints), dtype=np.int32
            )
        # Remove unwanted fields that come from the json file (if they exist)
        for k in ['date_captured', 'url', 'license']:
            if k in entry:
                del entry[k]
                
        entry['dataset_name'] = ''

        # add relationship annotations
        # sbj
        entry['sbj_gt_boxes'] = np.empty((0, 4), dtype=np.float32)
        entry['sbj_gt_classes'] = np.empty((0), dtype=np.int32)
        entry['sbj_gt_overlaps'] = scipy.sparse.csr_matrix(
            np.empty((0, self.num_obj_classes), dtype=np.float32)
        )
        # entry['sbj_box_to_gt_ind_map'] = np.empty((0), dtype=np.int32)
        # obj
        entry['obj_gt_boxes'] = np.empty((0, 4), dtype=np.float32)
        entry['obj_gt_classes'] = np.empty((0), dtype=np.int32)
        entry['obj_gt_overlaps'] = scipy.sparse.csr_matrix(
            np.empty((0, self.num_obj_classes), dtype=np.float32)
        )
        # entry['obj_box_to_gt_ind_map'] = np.empty((0), dtype=np.int32)
        # prd
        entry['prd_gt_classes'] = np.empty((0), dtype=np.int32)
        entry['prd_gt_overlaps'] = scipy.sparse.csr_matrix(
            np.empty((0, self.num_prd_classes), dtype=np.float32)
        )
        entry['pair_to_gt_ind_map'] = np.empty((0), dtype=np.int32)

    def _add_gt_annotations(self, entry):
        """Add ground truth annotation metadata to an roidb entry."""
        ann_ids = self.COCO.getAnnIds(imgIds=entry['id'], iscrowd=None)
        objs = self.COCO.loadAnns(ann_ids)
        # Sanitize bboxes -- some are invalid
        valid_objs = []
        valid_segms = []
        width = entry['width']
        height = entry['height']
        for obj in objs:
            if obj['area'] < cfg.TRAIN.GT_MIN_AREA:
                continue
            if 'ignore' in obj and obj['ignore'] == 1:
                continue
            # Convert form (x1, y1, w, h) to (x1, y1, x2, y2)
            x1, y1, x2, y2 = box_utils.xywh_to_xyxy(obj['bbox'])
            x1, y1, x2, y2 = box_utils.clip_xyxy_to_image(
                x1, y1, x2, y2, height, width
            )
            # Require non-zero seg area and more than 1x1 box size
            if obj['area'] > 0 and x2 > x1 and y2 > y1:
                obj['clean_bbox'] = [x1, y1, x2, y2]
                valid_objs.append(obj)
                # valid_segms.append(obj['segmentation'])
        num_valid_objs = len(valid_objs)

        boxes = np.zeros((num_valid_objs, 4), dtype=entry['boxes'].dtype)
        gt_classes = np.zeros((num_valid_objs), dtype=entry['gt_classes'].dtype)
        gt_overlaps = np.zeros(
            (num_valid_objs, self.num_classes),
            dtype=entry['gt_overlaps'].dtype
        )
        seg_areas = np.zeros((num_valid_objs), dtype=entry['seg_areas'].dtype)
        is_crowd = np.zeros((num_valid_objs), dtype=entry['is_crowd'].dtype)
        box_to_gt_ind_map = np.zeros(
            (num_valid_objs), dtype=entry['box_to_gt_ind_map'].dtype
        )
        if self.keypoints is not None:
            gt_keypoints = np.zeros(
                (num_valid_objs, 3, self.num_keypoints),
                dtype=entry['gt_keypoints'].dtype
            )

        im_has_visible_keypoints = False
        for ix, obj in enumerate(valid_objs):
            cls = self.json_category_id_to_contiguous_id[obj['category_id']]
            boxes[ix, :] = obj['clean_bbox']
            gt_classes[ix] = cls
            seg_areas[ix] = obj['area']
            is_crowd[ix] = obj['iscrowd']
            box_to_gt_ind_map[ix] = ix
            if self.keypoints is not None:
                gt_keypoints[ix, :, :] = self._get_gt_keypoints(obj)
                if np.sum(gt_keypoints[ix, 2, :]) > 0:
                    im_has_visible_keypoints = True
            if obj['iscrowd']:
                # Set overlap to -1 for all classes for crowd objects
                # so they will be excluded during training
                gt_overlaps[ix, :] = -1.0
            else:
                gt_overlaps[ix, cls] = 1.0
        entry['boxes'] = np.append(entry['boxes'], boxes, axis=0)
        entry['segms'].extend(valid_segms)
        entry['gt_classes'] = np.append(entry['gt_classes'], gt_classes)
        entry['seg_areas'] = np.append(entry['seg_areas'], seg_areas)
        entry['gt_overlaps'] = np.append(
            entry['gt_overlaps'].toarray(), gt_overlaps, axis=0
        )
        entry['gt_overlaps'] = scipy.sparse.csr_matrix(entry['gt_overlaps'])
        entry['is_crowd'] = np.append(entry['is_crowd'], is_crowd)
        entry['box_to_gt_ind_map'] = np.append(
            entry['box_to_gt_ind_map'], box_to_gt_ind_map
        )
        if self.keypoints is not None:
            entry['gt_keypoints'] = np.append(
                entry['gt_keypoints'], gt_keypoints, axis=0
            )
            entry['has_visible_keypoints'] = im_has_visible_keypoints
            
        entry['dataset_name'] = self.name

        # add relationship annotations
        im_rels = self.rel_anns[entry['file_name']]
        sbj_gt_boxes = np.zeros((len(im_rels), 4), dtype=entry['sbj_gt_boxes'].dtype)
        obj_gt_boxes = np.zeros((len(im_rels), 4), dtype=entry['obj_gt_boxes'].dtype)
        sbj_gt_classes = np.zeros(len(im_rels), dtype=entry['sbj_gt_classes'].dtype)
        obj_gt_classes = np.zeros(len(im_rels), dtype=entry['obj_gt_classes'].dtype)
        prd_gt_classes = np.zeros(len(im_rels), dtype=entry['prd_gt_classes'].dtype)
        for ix, rel in enumerate(im_rels):
            # sbj
            sbj_gt_box = box_utils_rel.y1y2x1x2_to_x1y1x2y2(rel['subject']['bbox'])
            sbj_gt_boxes[ix] = sbj_gt_box
            sbj_gt_classes[ix] = rel['subject']['category']  # excludes background
            # obj
            obj_gt_box = box_utils_rel.y1y2x1x2_to_x1y1x2y2(rel['object']['bbox'])
            obj_gt_boxes[ix] = obj_gt_box
            obj_gt_classes[ix] = rel['object']['category']  # excludes background
            # prd
            prd_gt_classes[ix] = rel['predicate']  # exclude background
        entry['sbj_gt_boxes'] = np.append(entry['sbj_gt_boxes'], sbj_gt_boxes, axis=0)
        entry['obj_gt_boxes'] = np.append(entry['obj_gt_boxes'], obj_gt_boxes, axis=0)
        entry['sbj_gt_classes'] = np.append(entry['sbj_gt_classes'], sbj_gt_classes)
        entry['obj_gt_classes'] = np.append(entry['obj_gt_classes'], obj_gt_classes)
        entry['prd_gt_classes'] = np.append(entry['prd_gt_classes'], prd_gt_classes)
        # misc
        sbj_gt_overlaps = np.zeros(
            (len(im_rels), self.num_obj_classes), dtype=entry['sbj_gt_overlaps'].dtype)
        for ix in range(len(im_rels)):
            sbj_cls = sbj_gt_classes[ix]
            sbj_gt_overlaps[ix, sbj_cls] = 1.0
        entry['sbj_gt_overlaps'] = np.append(
            entry['sbj_gt_overlaps'].toarray(), sbj_gt_overlaps, axis=0)
        entry['sbj_gt_overlaps'] = scipy.sparse.csr_matrix(entry['sbj_gt_overlaps'])

        obj_gt_overlaps = np.zeros(
            (len(im_rels), self.num_obj_classes), dtype=entry['obj_gt_overlaps'].dtype)
        for ix in range(len(im_rels)):
            obj_cls = obj_gt_classes[ix]
            obj_gt_overlaps[ix, obj_cls] = 1.0
        entry['obj_gt_overlaps'] = np.append(
            entry['obj_gt_overlaps'].toarray(), obj_gt_overlaps, axis=0)
        entry['obj_gt_overlaps'] = scipy.sparse.csr_matrix(entry['obj_gt_overlaps'])

        prd_gt_overlaps = np.zeros(
            (len(im_rels), self.num_prd_classes), dtype=entry['prd_gt_overlaps'].dtype)
        pair_to_gt_ind_map = np.zeros(
            (len(im_rels)), dtype=entry['pair_to_gt_ind_map'].dtype)
        for ix in range(len(im_rels)):
            prd_cls = prd_gt_classes[ix]
            prd_gt_overlaps[ix, prd_cls] = 1.0
            pair_to_gt_ind_map[ix] = ix
        entry['prd_gt_overlaps'] = np.append(
            entry['prd_gt_overlaps'].toarray(), prd_gt_overlaps, axis=0)
        entry['prd_gt_overlaps'] = scipy.sparse.csr_matrix(entry['prd_gt_overlaps'])
        entry['pair_to_gt_ind_map'] = np.append(
            entry['pair_to_gt_ind_map'], pair_to_gt_ind_map)
        
        for k in ['file_name']:
            if k in entry:
                del entry[k]

    def _add_gt_from_cache(self, roidb, cache_filepath):
        """Add ground truth annotation metadata from cached file."""
        logger.info('Loading cached gt_roidb from %s', cache_filepath)
        with open(cache_filepath, 'rb') as fp:
            cached_roidb = pickle.load(fp)

        assert len(roidb) == len(cached_roidb)

        for entry, cached_entry in zip(roidb, cached_roidb):
            values = [cached_entry[key] for key in self.valid_cached_keys]
            dataset_name, boxes, segms, gt_classes, seg_areas, gt_overlaps, is_crowd, box_to_gt_ind_map, \
                sbj_gt_boxes, sbj_gt_classes, obj_gt_boxes, obj_gt_classes, prd_gt_classes, \
                sbj_gt_overlaps, obj_gt_overlaps, prd_gt_overlaps, pair_to_gt_ind_map = values[:len(self.valid_cached_keys)]
            if self.keypoints is not None:
                gt_keypoints, has_visible_keypoints = values[len(self.valid_cached_keys):]
            entry['dataset_name'] = dataset_name
            entry['boxes'] = np.append(entry['boxes'], boxes, axis=0)
            entry['segms'].extend(segms)
            entry['gt_classes'] = np.append(entry['gt_classes'], gt_classes)
            entry['seg_areas'] = np.append(entry['seg_areas'], seg_areas)
            entry['gt_overlaps'] = scipy.sparse.csr_matrix(gt_overlaps)
            entry['is_crowd'] = np.append(entry['is_crowd'], is_crowd)
            entry['box_to_gt_ind_map'] = np.append(
                entry['box_to_gt_ind_map'], box_to_gt_ind_map
            )
            if self.keypoints is not None:
                entry['gt_keypoints'] = np.append(
                    entry['gt_keypoints'], gt_keypoints, axis=0
                )
                entry['has_visible_keypoints'] = has_visible_keypoints
                
            # add relationship annotations
            entry['sbj_gt_boxes'] = np.append(entry['sbj_gt_boxes'], sbj_gt_boxes, axis=0)
            entry['sbj_gt_classes'] = np.append(entry['sbj_gt_classes'], sbj_gt_classes)
            entry['sbj_gt_overlaps'] = scipy.sparse.csr_matrix(sbj_gt_overlaps)
            entry['obj_gt_boxes'] = np.append(entry['obj_gt_boxes'], obj_gt_boxes, axis=0)
            entry['obj_gt_classes'] = np.append(entry['obj_gt_classes'], obj_gt_classes)
            entry['obj_gt_overlaps'] = scipy.sparse.csr_matrix(obj_gt_overlaps)
            entry['prd_gt_classes'] = np.append(entry['prd_gt_classes'], prd_gt_classes)
            entry['prd_gt_overlaps'] = scipy.sparse.csr_matrix(prd_gt_overlaps)
            entry['pair_to_gt_ind_map'] = np.append(
                entry['pair_to_gt_ind_map'], pair_to_gt_ind_map)

    def _add_proposals_from_file(
        self, roidb, proposal_file, min_proposal_size, top_k, crowd_thresh
    ):
        """Add proposals from a proposals file to an roidb."""
        logger.info('Loading proposals from: {}'.format(proposal_file))
        with open(proposal_file, 'r') as f:
            proposals = pickle.load(f)
        id_field = 'indexes' if 'indexes' in proposals else 'ids'  # compat fix
        _sort_proposals(proposals, id_field)
        box_list = []
        for i, entry in enumerate(roidb):
            if i % 2500 == 0:
                logger.info(' {:d}/{:d}'.format(i + 1, len(roidb)))
            boxes = proposals['boxes'][i]
            # Sanity check that these boxes are for the correct image id
            assert entry['id'] == proposals[id_field][i]
            # Remove duplicate boxes and very small boxes and then take top k
            boxes = box_utils.clip_boxes_to_image(
                boxes, entry['height'], entry['width']
            )
            keep = box_utils.unique_boxes(boxes)
            boxes = boxes[keep, :]
            keep = box_utils.filter_small_boxes(boxes, min_proposal_size)
            boxes = boxes[keep, :]
            if top_k > 0:
                boxes = boxes[:top_k, :]
            box_list.append(boxes)
        _merge_proposal_boxes_into_roidb(roidb, box_list)
        if crowd_thresh > 0:
            _filter_crowd_proposals(roidb, crowd_thresh)

    def _init_keypoints(self):
        """Initialize COCO keypoint information."""
        self.keypoints = None
        self.keypoint_flip_map = None
        self.keypoints_to_id_map = None
        self.num_keypoints = 0
        # Thus far only the 'person' category has keypoints
        if 'person' in self.category_to_id_map:
            cat_info = self.COCO.loadCats([self.category_to_id_map['person']])
        else:
            return

        # Check if the annotations contain keypoint data or not
        if 'keypoints' in cat_info[0]:
            keypoints = cat_info[0]['keypoints']
            self.keypoints_to_id_map = dict(
                zip(keypoints, range(len(keypoints))))
            self.keypoints = keypoints
            self.num_keypoints = len(keypoints)
            if cfg.KRCNN.NUM_KEYPOINTS != -1:
                assert cfg.KRCNN.NUM_KEYPOINTS == self.num_keypoints, \
                    "number of keypoints should equal when using multiple datasets"
            else:
                cfg.KRCNN.NUM_KEYPOINTS = self.num_keypoints
            self.keypoint_flip_map = {
                'left_eye': 'right_eye',
                'left_ear': 'right_ear',
                'left_shoulder': 'right_shoulder',
                'left_elbow': 'right_elbow',
                'left_wrist': 'right_wrist',
                'left_hip': 'right_hip',
                'left_knee': 'right_knee',
                'left_ankle': 'right_ankle'}

    def _get_gt_keypoints(self, obj):
        """Return ground truth keypoints."""
        if 'keypoints' not in obj:
            return None
        kp = np.array(obj['keypoints'])
        x = kp[0::3]  # 0-indexed x coordinates
        y = kp[1::3]  # 0-indexed y coordinates
        # 0: not labeled; 1: labeled, not inside mask;
        # 2: labeled and inside mask
        v = kp[2::3]
        num_keypoints = len(obj['keypoints']) / 3
        assert num_keypoints == self.num_keypoints
        gt_kps = np.ones((3, self.num_keypoints), dtype=np.int32)
        for i in range(self.num_keypoints):
            gt_kps[0, i] = x[i]
            gt_kps[1, i] = y[i]
            gt_kps[2, i] = v[i]
        return gt_kps


def add_rel_proposals(roidb, sbj_rois, obj_rois, det_rois, scales):
    """Add proposal boxes (rois) to an roidb that has ground-truth annotations
    but no proposals. If the proposals are not at the original image scale,
    specify the scale factor that separate them in scales.
    """
    assert (sbj_rois[:, 0] == obj_rois[:, 0]).all()
    sbj_box_list = []
    obj_box_list = []
    for i, entry in enumerate(roidb):
        inv_im_scale = 1. / scales[i]
        idx = np.where(sbj_rois[:, 0] == i)[0]
        # include pairs where at least one box is gt
        det_idx = np.where(det_rois[:, 0] == i)[0]
        im_det_boxes = det_rois[det_idx, 1:] * inv_im_scale
        sbj_gt_boxes = entry['sbj_gt_boxes']
        obj_gt_boxes = entry['obj_gt_boxes']
        unique_sbj_gt_boxes = np.unique(sbj_gt_boxes, axis=0)
        unique_obj_gt_boxes = np.unique(obj_gt_boxes, axis=0)
        # sbj_gt w/ obj_det
        sbj_gt_boxes_paired_w_det = np.repeat(unique_sbj_gt_boxes, im_det_boxes.shape[0], axis=0)
        obj_det_boxes_paired_w_gt = np.tile(im_det_boxes, (unique_sbj_gt_boxes.shape[0], 1))
        # sbj_det w/ obj_gt
        sbj_det_boxes_paired_w_gt = np.repeat(im_det_boxes, unique_obj_gt_boxes.shape[0], axis=0)
        obj_gt_boxes_paired_w_det = np.tile(unique_obj_gt_boxes, (im_det_boxes.shape[0], 1))
        # sbj_gt w/ obj_gt
        sbj_gt_boxes_paired_w_gt = np.repeat(unique_sbj_gt_boxes, unique_obj_gt_boxes.shape[0], axis=0)
        obj_gt_boxes_paired_w_gt = np.tile(unique_obj_gt_boxes, (unique_sbj_gt_boxes.shape[0], 1))
        # now concatenate them all
        sbj_box_list.append(np.concatenate(
            (sbj_rois[idx, 1:] * inv_im_scale, sbj_gt_boxes_paired_w_det, sbj_det_boxes_paired_w_gt, sbj_gt_boxes_paired_w_gt)))
        obj_box_list.append(np.concatenate(
            (obj_rois[idx, 1:] * inv_im_scale, obj_det_boxes_paired_w_gt, obj_gt_boxes_paired_w_det, obj_gt_boxes_paired_w_gt)))
        _merge_paired_boxes_into_roidb(roidb, sbj_box_list, obj_box_list)
        _add_prd_class_assignments(roidb)
    
    
def _merge_paired_boxes_into_roidb(roidb, sbj_box_list, obj_box_list):
    assert len(sbj_box_list) == len(obj_box_list) == len(roidb)
    for i, entry in enumerate(roidb):
        sbj_boxes = sbj_box_list[i]
        obj_boxes = obj_box_list[i]
        assert sbj_boxes.shape[0] == obj_boxes.shape[0]
        num_pairs = sbj_boxes.shape[0]
        sbj_gt_overlaps = np.zeros(
            (num_pairs, entry['sbj_gt_overlaps'].shape[1]),
            dtype=entry['sbj_gt_overlaps'].dtype
        )
        obj_gt_overlaps = np.zeros(
            (num_pairs, entry['obj_gt_overlaps'].shape[1]),
            dtype=entry['obj_gt_overlaps'].dtype
        )
        prd_gt_overlaps = np.zeros(
            (num_pairs, entry['prd_gt_overlaps'].shape[1]),
            dtype=entry['prd_gt_overlaps'].dtype
        )
        pair_to_gt_ind_map = -np.ones(
            (num_pairs), dtype=entry['pair_to_gt_ind_map'].dtype
        )
        
        pair_gt_inds = np.arange(entry['prd_gt_classes'].shape[0])
        if len(pair_gt_inds) > 0:
            sbj_gt_boxes = entry['sbj_gt_boxes'][pair_gt_inds, :]
            sbj_gt_classes = entry['sbj_gt_classes'][pair_gt_inds]
            obj_gt_boxes = entry['obj_gt_boxes'][pair_gt_inds, :]
            obj_gt_classes = entry['obj_gt_classes'][pair_gt_inds]
            prd_gt_classes = entry['prd_gt_classes'][pair_gt_inds]
            sbj_to_gt_overlaps = box_utils.bbox_overlaps(
                sbj_boxes.astype(dtype=np.float32, copy=False),
                sbj_gt_boxes.astype(dtype=np.float32, copy=False)
            )
            obj_to_gt_overlaps = box_utils.bbox_overlaps(
                obj_boxes.astype(dtype=np.float32, copy=False),
                obj_gt_boxes.astype(dtype=np.float32, copy=False)
            )
            pair_to_gt_overlaps = np.minimum(sbj_to_gt_overlaps, obj_to_gt_overlaps)
            # Gt box that overlaps each input box the most
            # (ties are broken arbitrarily by class order)
            sbj_argmaxes = sbj_to_gt_overlaps.argmax(axis=1)
            sbj_maxes = sbj_to_gt_overlaps.max(axis=1)  # Amount of that overlap
            sbj_I = np.where(sbj_maxes >= 0)[0]  # Those boxes with non-zero overlap with gt boxes, get all items
            
            obj_argmaxes = obj_to_gt_overlaps.argmax(axis=1)
            obj_maxes = obj_to_gt_overlaps.max(axis=1)  # Amount of that overlap
            obj_I = np.where(obj_maxes >= 0)[0]  # Those boxes with non-zero overlap with gt boxes, get all items
            
            pair_argmaxes = pair_to_gt_overlaps.argmax(axis=1)
            pair_maxes = pair_to_gt_overlaps.max(axis=1)  # Amount of that overlap
            pair_I = np.where(pair_maxes >= 0)[0]  # Those boxes with non-zero overlap with gt boxes, get all items
            # Record max overlaps with the class of the appropriate gt box
            sbj_gt_overlaps[sbj_I, sbj_gt_classes[sbj_argmaxes[sbj_I]]] = sbj_maxes[sbj_I]
            obj_gt_overlaps[obj_I, obj_gt_classes[obj_argmaxes[obj_I]]] = obj_maxes[obj_I]
            prd_gt_overlaps[pair_I, prd_gt_classes[pair_argmaxes[pair_I]]] = pair_maxes[pair_I]
            pair_to_gt_ind_map[pair_I] = pair_gt_inds[pair_argmaxes[pair_I]]
        entry['sbj_boxes'] = sbj_boxes.astype(entry['sbj_gt_boxes'].dtype, copy=False)
        entry['sbj_gt_overlaps'] = sbj_gt_overlaps
        entry['sbj_gt_overlaps'] = scipy.sparse.csr_matrix(entry['sbj_gt_overlaps'])

        entry['obj_boxes'] = obj_boxes.astype(entry['obj_gt_boxes'].dtype, copy=False)
        entry['obj_gt_overlaps'] = obj_gt_overlaps
        entry['obj_gt_overlaps'] = scipy.sparse.csr_matrix(entry['obj_gt_overlaps'])

        entry['prd_gt_classes'] = -np.ones((num_pairs), dtype=entry['prd_gt_classes'].dtype)
        entry['prd_gt_overlaps'] = prd_gt_overlaps
        entry['prd_gt_overlaps'] = scipy.sparse.csr_matrix(entry['prd_gt_overlaps'])
        entry['pair_to_gt_ind_map'] = pair_to_gt_ind_map.astype(
                entry['pair_to_gt_ind_map'].dtype, copy=False)


def _add_prd_class_assignments(roidb):
    """Compute object category assignment for each box associated with each
    roidb entry.
    """
    for entry in roidb:
        sbj_gt_overlaps = entry['sbj_gt_overlaps'].toarray()
        max_sbj_overlaps = sbj_gt_overlaps.max(axis=1)
        max_sbj_classes = sbj_gt_overlaps.argmax(axis=1)
        entry['max_sbj_classes'] = max_sbj_classes
        entry['max_sbj_overlaps'] = max_sbj_overlaps
        
        obj_gt_overlaps = entry['obj_gt_overlaps'].toarray()
        max_obj_overlaps = obj_gt_overlaps.max(axis=1)
        max_obj_classes = obj_gt_overlaps.argmax(axis=1)
        entry['max_obj_classes'] = max_obj_classes
        entry['max_obj_overlaps'] = max_obj_overlaps

        prd_gt_overlaps = entry['prd_gt_overlaps'].toarray()
        max_pair_overlaps = prd_gt_overlaps.max(axis=1)
        max_prd_classes = prd_gt_overlaps.argmax(axis=1)
        entry['max_prd_classes'] = max_prd_classes
        entry['max_pair_overlaps'] = max_pair_overlaps
        # sanity checks
        # if max overlap is 0, the class must be background (class 0)
        # zero_inds = np.where(max_pair_overlaps == 0)[0]
        # assert all(max_prd_classes[zero_inds] == 0)
        # # if max overlap > 0, the class must be a fg class (not class 0)
        # nonzero_inds = np.where(max_pair_overlaps > 0)[0]
        # assert all(max_prd_classes[nonzero_inds] != 0)
        

def add_proposals(roidb, rois, scales, crowd_thresh):
    """Add proposal boxes (rois) to an roidb that has ground-truth annotations
    but no proposals. If the proposals are not at the original image scale,
    specify the scale factor that separate them in scales.
    """
    box_list = []
    for i in range(len(roidb)):
        inv_im_scale = 1. / scales[i]
        idx = np.where(rois[:, 0] == i)[0]
        box_list.append(rois[idx, 1:] * inv_im_scale)
    _merge_proposal_boxes_into_roidb(roidb, box_list)
    if crowd_thresh > 0:
        _filter_crowd_proposals(roidb, crowd_thresh)
    _add_class_assignments(roidb)


def _merge_proposal_boxes_into_roidb(roidb, box_list):
    """Add proposal boxes to each roidb entry."""
    assert len(box_list) == len(roidb)
    for i, entry in enumerate(roidb):
        boxes = box_list[i]
        num_boxes = boxes.shape[0]
        gt_overlaps = np.zeros(
            (num_boxes, entry['gt_overlaps'].shape[1]),
            dtype=entry['gt_overlaps'].dtype
        )
        box_to_gt_ind_map = -np.ones(
            (num_boxes), dtype=entry['box_to_gt_ind_map'].dtype
        )

        # Note: unlike in other places, here we intentionally include all gt
        # rois, even ones marked as crowd. Boxes that overlap with crowds will
        # be filtered out later (see: _filter_crowd_proposals).
        gt_inds = np.where(entry['gt_classes'] > 0)[0]
        if len(gt_inds) > 0:
            gt_boxes = entry['boxes'][gt_inds, :]
            gt_classes = entry['gt_classes'][gt_inds]
            proposal_to_gt_overlaps = box_utils.bbox_overlaps(
                boxes.astype(dtype=np.float32, copy=False),
                gt_boxes.astype(dtype=np.float32, copy=False)
            )
            # Gt box that overlaps each input box the most
            # (ties are broken arbitrarily by class order)
            argmaxes = proposal_to_gt_overlaps.argmax(axis=1)
            # Amount of that overlap
            maxes = proposal_to_gt_overlaps.max(axis=1)
            # Those boxes with non-zero overlap with gt boxes
            I = np.where(maxes > 0)[0]
            # Record max overlaps with the class of the appropriate gt box
            gt_overlaps[I, gt_classes[argmaxes[I]]] = maxes[I]
            box_to_gt_ind_map[I] = gt_inds[argmaxes[I]]
        entry['boxes'] = np.append(
            entry['boxes'],
            boxes.astype(entry['boxes'].dtype, copy=False),
            axis=0
        )
        entry['gt_classes'] = np.append(
            entry['gt_classes'],
            np.zeros((num_boxes), dtype=entry['gt_classes'].dtype)
        )
        entry['seg_areas'] = np.append(
            entry['seg_areas'],
            np.zeros((num_boxes), dtype=entry['seg_areas'].dtype)
        )
        entry['gt_overlaps'] = np.append(
            entry['gt_overlaps'].toarray(), gt_overlaps, axis=0
        )
        entry['gt_overlaps'] = scipy.sparse.csr_matrix(entry['gt_overlaps'])
        entry['is_crowd'] = np.append(
            entry['is_crowd'],
            np.zeros((num_boxes), dtype=entry['is_crowd'].dtype)
        )
        entry['box_to_gt_ind_map'] = np.append(
            entry['box_to_gt_ind_map'],
            box_to_gt_ind_map.astype(
                entry['box_to_gt_ind_map'].dtype, copy=False
            )
        )


def _filter_crowd_proposals(roidb, crowd_thresh):
    """Finds proposals that are inside crowd regions and marks them as
    overlap = -1 with each ground-truth rois, which means they will be excluded
    from training.
    """
    for entry in roidb:
        gt_overlaps = entry['gt_overlaps'].toarray()
        crowd_inds = np.where(entry['is_crowd'] == 1)[0]
        non_gt_inds = np.where(entry['gt_classes'] == 0)[0]
        if len(crowd_inds) == 0 or len(non_gt_inds) == 0:
            continue
        crowd_boxes = box_utils.xyxy_to_xywh(entry['boxes'][crowd_inds, :])
        non_gt_boxes = box_utils.xyxy_to_xywh(entry['boxes'][non_gt_inds, :])
        iscrowd_flags = [int(True)] * len(crowd_inds)
        ious = COCOmask.iou(non_gt_boxes, crowd_boxes, iscrowd_flags)
        bad_inds = np.where(ious.max(axis=1) > crowd_thresh)[0]
        gt_overlaps[non_gt_inds[bad_inds], :] = -1
        entry['gt_overlaps'] = scipy.sparse.csr_matrix(gt_overlaps)


def _add_class_assignments(roidb):
    """Compute object category assignment for each box associated with each
    roidb entry.
    """
    for entry in roidb:
        gt_overlaps = entry['gt_overlaps'].toarray()
        # max overlap with gt over classes (columns)
        max_overlaps = gt_overlaps.max(axis=1)
        # gt class that had the max overlap
        max_classes = gt_overlaps.argmax(axis=1)
        entry['max_classes'] = max_classes
        entry['max_overlaps'] = max_overlaps
        # sanity checks
        # if max overlap is 0, the class must be background (class 0)
        zero_inds = np.where(max_overlaps == 0)[0]
        assert all(max_classes[zero_inds] == 0)
        # if max overlap > 0, the class must be a fg class (not class 0)
        nonzero_inds = np.where(max_overlaps > 0)[0]
        assert all(max_classes[nonzero_inds] != 0)


def _sort_proposals(proposals, id_field):
    """Sort proposals by the specified id field."""
    order = np.argsort(proposals[id_field])
    fields_to_sort = ['boxes', id_field, 'scores']
    for k in fields_to_sort:
        proposals[k] = [proposals[k][i] for i in order]
