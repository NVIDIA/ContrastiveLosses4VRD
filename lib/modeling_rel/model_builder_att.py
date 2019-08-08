# Adapted from Detectron.pytorch/lib/modeling/model_builder.py
# for this project by Ji Zhang, 2019

from functools import wraps
import importlib
import logging
import numpy as np
import copy
import json

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from core.config import cfg
from model.roi_layers import ROIPool, ROIAlign
# from model.roi_pooling.functions.roi_pool import RoIPoolFunction
# from model.roi_crop.functions.roi_crop import RoICropFunction
# from modeling.roi_xfrom.roi_align.functions.roi_align import RoIAlignFunction
import modeling.rpn_heads as rpn_heads
import modeling_rel.fast_rcnn_heads as fast_rcnn_heads
import modeling_rel.attpn_heads as attpn_heads
import modeling_rel.attdn_heads as attdn_heads
import utils.boxes as box_utils
import utils.blob as blob_utils
import utils_rel.net_rel as net_utils_rel
import utils.resnet_weights_helper as resnet_utils

logger = logging.getLogger(__name__)


def get_func(func_name):
    """Helper to return a function object by name. func_name must identify a
    function in this module or the path to a function relative to the base
    'modeling' module.
    """
    if func_name == '':
        return None
    try:
        # these two keywords means we need to use the functions from the modeling_rel directory
        if func_name.find('VGG') >= 0 or func_name.find('roi_2mlp_head') >= 0:
            dir_name = 'modeling_rel.'
        else:
            dir_name = 'modeling.'
        parts = func_name.split('.')
        # Refers to a function in this module
        if len(parts) == 1:
            return globals()[parts[0]]
        # Otherwise, assume we're referencing a module under modeling
        module_name = dir_name + '.'.join(parts[:-1])
        module = importlib.import_module(module_name)
        return getattr(module, parts[-1])
    except Exception:
        logger.error('Failed to find function: %s', func_name)
        raise


def check_inference(net_func):
    @wraps(net_func)
    def wrapper(self, *args, **kwargs):
        if not self.training:
            if cfg.PYTORCH_VERSION_LESS_THAN_040:
                return net_func(self, *args, **kwargs)
            else:
                with torch.no_grad():
                    return net_func(self, *args, **kwargs)
        else:
            raise ValueError('You should call this function only on inference.'
                              'Set the network in inference mode by net.eval().')

    return wrapper


class Generalized_RCNN(nn.Module):
    def __init__(self):
        super().__init__()

        # For cache
        self.mapping_to_detectron = None
        self.orphans_in_detectron = None

        # Backbone for feature extraction
        self.Conv_Body = get_func(cfg.MODEL.CONV_BODY)()

        # Region Proposal Network
        if cfg.RPN.RPN_ON:
            self.RPN = rpn_heads.generic_rpn_outputs(
                self.Conv_Body.dim_out, self.Conv_Body.spatial_scale)

        if cfg.FPN.FPN_ON:
            # Only supports case when RPN and ROI min levels are the same
            assert cfg.FPN.RPN_MIN_LEVEL == cfg.FPN.ROI_MIN_LEVEL
            # RPN max level can be >= to ROI max level
            assert cfg.FPN.RPN_MAX_LEVEL >= cfg.FPN.ROI_MAX_LEVEL
            # FPN RPN max level might be > FPN ROI max level in which case we
            # need to discard some leading conv blobs (blobs are ordered from
            # max/coarsest level to min/finest level)
            self.num_roi_levels = cfg.FPN.ROI_MAX_LEVEL - cfg.FPN.ROI_MIN_LEVEL + 1

            # Retain only the spatial scales that will be used for RoI heads. `Conv_Body.spatial_scale`
            # may include extra scales that are used for RPN proposals, but not for RoI heads.
            self.Conv_Body.spatial_scale = self.Conv_Body.spatial_scale[-self.num_roi_levels:]

        # BBOX Branch
        self.Box_Head = get_func(cfg.FAST_RCNN.ROI_BOX_HEAD)(
            self.RPN.dim_out, self.roi_feature_transform, self.Conv_Body.spatial_scale)
        self.Box_Outs = fast_rcnn_heads.fast_rcnn_outputs(
            self.Box_Head.dim_out)
            
        self.Att_RCNN = copy.deepcopy(self)
        del self.Att_RCNN.RPN
        del self.Att_RCNN.Box_Outs
        
        # AttPN
        self.AttPN = attpn_heads.generic_attpn_outputs()
        # AttDN
        self.AttDN = attdn_heads.attdn_head(self.Box_Head.dim_out)

        self._init_modules()

    def _init_modules(self):
        # VGG16 imagenet pretrained model is initialized in VGG16.py
        if cfg.RESNETS.IMAGENET_PRETRAINED_WEIGHTS != '':
            logger.info("Loading pretrained weights from %s", cfg.RESNETS.IMAGENET_PRETRAINED_WEIGHTS)
            resnet_utils.load_pretrained_imagenet_weights(self)
                
        if cfg.RESNETS.VRD_PRETRAINED_WEIGHTS != '':
            self.load_detector_weights(cfg.RESNETS.VRD_PRETRAINED_WEIGHTS)
            
        if cfg.RESNETS.VG_PRETRAINED_WEIGHTS != '':
            self.load_detector_weights(cfg.RESNETS.VG_PRETRAINED_WEIGHTS)
            
        if cfg.RESNETS.OI_REL_PRETRAINED_WEIGHTS != '':
            self.load_detector_weights(cfg.RESNETS.OI_REL_PRETRAINED_WEIGHTS)

        if cfg.TRAIN.FREEZE_CONV_BODY:
            for p in self.Conv_Body.parameters():
                p.requires_grad = False
                
        # By Ji on 05/11/2019
        if cfg.RESNETS.ATT_RCNN_PRETRAINED_WEIGHTS != '':
            logger.info("loading att_rcnn pretrained weights from %s", cfg.RESNETS.ATT_RCNN_PRETRAINED_WEIGHTS)
            checkpoint = torch.load(cfg.RESNETS.ATT_RCNN_PRETRAINED_WEIGHTS, map_location=lambda storage, loc: storage)
            to_be_deleted = []
            for p, _ in checkpoint['model'].items():
                if p.find('Att_RCNN') < 0 or p.find('Box_Outs') >= 0:
                    to_be_deleted.append(p)
            for p in to_be_deleted:
                del checkpoint['model'][p]
            net_utils_rel.load_ckpt_rel(self.Att_RCNN, checkpoint['model'])
                    
#         if cfg.RESNETS.OI_ATT_PRETRAINED_WEIGHTS != '':  # this means dataset contains 'oi'
#             logger.info("loading att pretrained weights from %s", cfg.RESNETS.OI_ATT_PRETRAINED_WEIGHTS)
#             checkpoint = torch.load(cfg.RESNETS.OI_ATT_PRETRAINED_WEIGHTS, map_location=lambda storage, loc: storage)
#             # not using the last softmax layers
#             del checkpoint['model']['Box_Outs.cls_score.weight']
#             del checkpoint['model']['Box_Outs.cls_score.bias']
#             del checkpoint['model']['Box_Outs.bbox_pred.weight']
#             del checkpoint['model']['Box_Outs.bbox_pred.bias']
#             net_utils_rel.load_ckpt_rel(self.Att_RCNN, checkpoint['model'])
    
    def load_detector_weights(self, weight_name):
        logger.info("loading pretrained weights from %s", weight_name)
        checkpoint = torch.load(weight_name, map_location=lambda storage, loc: storage)
        net_utils_rel.load_ckpt_rel(self, checkpoint['model'])
        # freeze everything above the att module
        for p in self.Conv_Body.parameters():
            p.requires_grad = False
        for p in self.RPN.parameters():
            p.requires_grad = False
        if not cfg.MODEL.UNFREEZE_DET:
            for p in self.Box_Head.parameters():
                p.requires_grad = False
            for p in self.Box_Outs.parameters():
                p.requires_grad = False

    def forward(self, data, im_info, dataset_name=None, roidb=None, **rpn_kwargs):
        if cfg.PYTORCH_VERSION_LESS_THAN_040:
            return self._forward(data, im_info, dataset_name, roidb, **rpn_kwargs)
        else:
            with torch.set_grad_enabled(self.training):
                return self._forward(data, im_info, dataset_name, roidb, **rpn_kwargs)

    def _forward(self, data, im_info, dataset_name=None, roidb=None, **rpn_kwargs):
        im_data = data
        if self.training:
            roidb = list(map(lambda x: blob_utils.deserialize(x)[0], roidb))
        if dataset_name is not None:
            dataset_name = blob_utils.deserialize(dataset_name)
        else:
            dataset_name = cfg.TRAIN.DATASETS[0] if self.training else cfg.TEST.DATASETS[0]  # assuming only one dataset per run

        device_id = im_data.get_device()

        return_dict = {}  # A dict to collect return variables

        blob_conv = self.Conv_Body(im_data)
        blob_conv_att = self.Att_RCNN.Conv_Body(im_data)

        rpn_ret = self.RPN(blob_conv, im_info, roidb)

        if cfg.FPN.FPN_ON:
            # Retain only the blobs that will be used for RoI heads. `blob_conv` may include
            # extra blobs that are used for RPN proposals, but not for RoI heads.
            blob_conv = blob_conv[-self.num_roi_levels:]
            blob_conv_att = blob_conv_att[-self.num_roi_levels:]
            if dataset_name.find('oi') >= 0:
                blob_conv_att = blob_conv_att[-self.num_roi_levels:]

        if not self.training:
            return_dict['blob_conv'] = blob_conv

        if cfg.MODEL.SHARE_RES5 and self.training:
            box_feat, res5_feat = self.Box_Head(blob_conv, rpn_ret, use_relu=True)
        else:
            box_feat = self.Box_Head(blob_conv, rpn_ret, use_relu=True)
        cls_score, bbox_pred = self.Box_Outs(box_feat)
        
        # now go through the predicate branch
        if self.training:
            fg_inds = np.where(rpn_ret['labels_int32'] > 0)[0]
            det_rois = rpn_ret['rois'][fg_inds]
            det_labels = rpn_ret['labels_int32'][fg_inds]
            det_scores = F.softmax(cls_score[fg_inds], dim=1)
            att_ret = self.AttPN(det_rois, det_labels, det_scores, im_info, dataset_name, roidb)
        else:
            score_thresh = cfg.TEST.SCORE_THRESH
            while score_thresh >= -1e-06:  # a negative value very close to 0.0
                det_rois, det_labels, det_scores = \
                    self.prepare_det_rois(rpn_ret['rois'], cls_score, bbox_pred, im_info, score_thresh)
                att_ret = self.AttPN(det_rois, det_labels, det_scores, im_info, dataset_name, roidb)
                valid_len = len(att_ret['obj_rois'])
                if valid_len > 0:
                    break
                logger.info('Got {} att_rois when score_thresh={}, changing to {}'.format(
                    valid_len, score_thresh, score_thresh - 0.01))
                score_thresh -= 0.01
            if len(att_ret['obj_rois']) == 0:
                return_dict['obj_rois'] = None
                return_dict['obj_labels'] = None
                return_dict['obj_scores'] = None
                return_dict['att_scores'] = None
                return return_dict

        if cfg.MODEL.NO_FC7_RELU:
            use_relu = False
        else:
            use_relu = True

        att_feat = self.Att_RCNN.Box_Head(blob_conv_att, att_ret, rois_name='obj_rois', use_relu=use_relu)
        if cfg.MODEL.USE_FREQ_BIAS:
            obj_labels = att_ret['all_obj_labels_int32']
        else:
            obj_labels = None
        att_cls_scores = self.AttDN(att_feat, obj_labels)  # obj_labels start from 0

        if self.training:
            return_dict['losses'] = {}
            return_dict['metrics'] = {}
            # rpn loss
            rpn_kwargs.update(dict(
                (k, rpn_ret[k]) for k in rpn_ret.keys()
                if (k.startswith('rpn_cls_logits') or k.startswith('rpn_bbox_pred'))
            ))
            loss_rpn_cls, loss_rpn_bbox = rpn_heads.generic_rpn_losses(**rpn_kwargs)
            if cfg.FPN.FPN_ON:
                for i, lvl in enumerate(range(cfg.FPN.RPN_MIN_LEVEL, cfg.FPN.RPN_MAX_LEVEL + 1)):
                    return_dict['losses']['loss_rpn_cls_fpn%d' % lvl] = loss_rpn_cls[i]
                    return_dict['losses']['loss_rpn_bbox_fpn%d' % lvl] = loss_rpn_bbox[i]
            else:
                return_dict['losses']['loss_rpn_cls'] = loss_rpn_cls
                return_dict['losses']['loss_rpn_bbox'] = loss_rpn_bbox
            # bbox loss
            loss_cls, loss_bbox, accuracy_cls = fast_rcnn_heads.fast_rcnn_losses(
                cls_score, bbox_pred, rpn_ret['labels_int32'], rpn_ret['bbox_targets'],
                rpn_ret['bbox_inside_weights'], rpn_ret['bbox_outside_weights'])
            return_dict['losses']['loss_cls'] = loss_cls
            return_dict['losses']['loss_bbox'] = loss_bbox
            return_dict['metrics']['accuracy_cls'] = accuracy_cls
            
            loss_cls_att, accuracy_cls_att = attdn_heads.attdn_losses(att_cls_scores, att_ret['all_att_labels_int32'])
            
            return_dict['losses']['loss_cls_att'] = loss_cls_att
            return_dict['metrics']['accuracy_cls_att'] = accuracy_cls_att
                
            # pytorch0.4 bug on gathering scalar(0-dim) tensors
            for k, v in return_dict['losses'].items():
                return_dict['losses'][k] = v.unsqueeze(0)
            for k, v in return_dict['metrics'].items():
                return_dict['metrics'][k] = v.unsqueeze(0)
        else:
            # Testing
            return_dict['obj_rois'] = att_ret['obj_rois']
            return_dict['obj_labels'] = att_ret['obj_labels']  # start from 1
            return_dict['obj_scores'] = att_ret['obj_scores']
            return_dict['att_scores'] = att_cls_scores

        return return_dict
    
    def get_roi_inds(self, det_labels, lbls):
        lbl_set = np.array(lbls)
        inds = np.where(np.isin(det_labels, lbl_set))[0]
        return inds
    
    def prepare_det_rois(self, rois, cls_scores, bbox_pred, im_info, score_thresh=cfg.TEST.SCORE_THRESH):
        im_info = im_info.data.cpu().numpy()
        # NOTE: 'rois' is numpy array while
        # 'cls_scores' and 'bbox_pred' are pytorch tensors
        scores = cls_scores.data.cpu().numpy().squeeze()
        # Apply bounding-box regression deltas
        box_deltas = bbox_pred.data.cpu().numpy().squeeze()
        
        assert rois.shape[0] == scores.shape[0] == box_deltas.shape[0]
        
        det_rois = np.empty((0, 5), dtype=np.float32)
        det_labels = np.empty((0), dtype=np.float32)
        det_scores = np.empty((0), dtype=np.float32)
        for im_i in range(cfg.TRAIN.IMS_PER_BATCH):
            # get all boxes that belong to this image
            inds = np.where(abs(rois[:, 0] - im_i) < 1e-06)[0]
            # unscale back to raw image space
            im_boxes = rois[inds, 1:5] / im_info[im_i, 2]
            im_scores = scores[inds]
            # In case there is 1 proposal
            im_scores = im_scores.reshape([-1, im_scores.shape[-1]])
            # In case there is 1 proposal
            im_box_deltas = box_deltas[inds]
            im_box_deltas = im_box_deltas.reshape([-1, im_box_deltas[inds].shape[-1]])

            im_scores, im_boxes = self.get_det_boxes(im_boxes, im_scores, im_box_deltas, im_info[im_i][:2] / im_info[im_i][2])
            im_scores, im_boxes, im_labels = self.box_results_with_nms_and_limit(im_scores, im_boxes, score_thresh)
            
            batch_inds = im_i * np.ones(
                (im_boxes.shape[0], 1), dtype=np.float32)
            
            im_det_rois = np.hstack((batch_inds, im_boxes * im_info[im_i, 2]))
            det_rois = np.append(det_rois, im_det_rois, axis=0)
            
            det_labels = np.append(det_labels, im_labels, axis=0)
            
            det_scores = np.append(det_scores, im_scores, axis=0)
        
        return det_rois, det_labels, det_scores

    def get_det_boxes(self, boxes, scores, box_deltas, h_and_w):

        if cfg.TEST.BBOX_REG:
            if cfg.MODEL.CLS_AGNOSTIC_BBOX_REG:
                # Remove predictions for bg class (compat with MSRA code)
                box_deltas = box_deltas[:, -4:]
            if cfg.TRAIN.BBOX_NORMALIZE_TARGETS_PRECOMPUTED:
                # (legacy) Optionally normalize targets by a precomputed mean and stdev
                box_deltas = box_deltas.view(-1, 4) * cfg.TRAIN.BBOX_NORMALIZE_STDS \
                             + cfg.TRAIN.BBOX_NORMALIZE_MEANS
            pred_boxes = box_utils.bbox_transform(boxes, box_deltas, cfg.MODEL.BBOX_REG_WEIGHTS)
            pred_boxes = box_utils.clip_tiled_boxes(pred_boxes, h_and_w)
            if cfg.MODEL.CLS_AGNOSTIC_BBOX_REG:
                pred_boxes = np.tile(pred_boxes, (1, scores.shape[1]))
        else:
            # Simply repeat the boxes, once for each class
            pred_boxes = np.tile(boxes, (1, scores.shape[1]))

        if cfg.DEDUP_BOXES > 0 and not cfg.MODEL.FASTER_RCNN:
            # Map scores and predictions back to the original set of boxes
            scores = scores[inv_index, :]
            pred_boxes = pred_boxes[inv_index, :]
            
        return scores, pred_boxes
    
    def box_results_with_nms_and_limit(self, scores, boxes, score_thresh=cfg.TEST.SCORE_THRESH):
        num_classes = cfg.MODEL.NUM_CLASSES
        cls_boxes = [[] for _ in range(num_classes)]
        # Apply threshold on detection probabilities and apply NMS
        # Skip j = 0, because it's the background class
        for j in range(1, num_classes):
            inds = np.where(scores[:, j] > score_thresh)[0]
            scores_j = scores[inds, j]
            boxes_j = boxes[inds, j * 4:(j + 1) * 4]
            dets_j = np.hstack((boxes_j, scores_j[:, np.newaxis])).astype(np.float32, copy=False)
            if cfg.TEST.SOFT_NMS.ENABLED:
                nms_dets, _ = box_utils.soft_nms(
                    dets_j,
                    sigma=cfg.TEST.SOFT_NMS.SIGMA,
                    overlap_thresh=cfg.TEST.NMS,
                    score_thresh=0.0001,
                    method=cfg.TEST.SOFT_NMS.METHOD
                )
            else:
                keep = box_utils.nms(dets_j, cfg.TEST.NMS)
                nms_dets = dets_j[keep, :]
            # add labels
            label_j = np.ones((nms_dets.shape[0], 1), dtype=np.float32) * j
            nms_dets = np.hstack((nms_dets, label_j))
            # Refine the post-NMS boxes using bounding-box voting
            if cfg.TEST.BBOX_VOTE.ENABLED:
                nms_dets = box_utils.box_voting(
                    nms_dets,
                    dets_j,
                    cfg.TEST.BBOX_VOTE.VOTE_TH,
                    scoring_method=cfg.TEST.BBOX_VOTE.SCORING_METHOD
                )
            cls_boxes[j] = nms_dets

        # Limit to max_per_image detections **over all classes**
        if cfg.TEST.DETECTIONS_PER_IM > 0:
            image_scores = np.hstack(
                [cls_boxes[j][:, -2] for j in range(1, num_classes)]
            )
            if len(image_scores) > cfg.TEST.DETECTIONS_PER_IM:
                image_thresh = np.sort(image_scores)[-cfg.TEST.DETECTIONS_PER_IM]
                for j in range(1, num_classes):
                    keep = np.where(cls_boxes[j][:, -2] >= image_thresh)[0]
                    cls_boxes[j] = cls_boxes[j][keep, :]

        im_results = np.vstack([cls_boxes[j] for j in range(1, num_classes)])
        boxes = im_results[:, :-2]
        scores = im_results[:, -2]
        labels = im_results[:, -1]

        return scores, boxes, labels

    def roi_feature_transform(self, blobs_in, rpn_ret, blob_rois='rois', method='RoIPoolF',
                              resolution=7, spatial_scale=1. / 16., sampling_ratio=0):
        """Add the specified RoI pooling method. The sampling_ratio argument
        is supported for some, but not all, RoI transform methods.

        RoIFeatureTransform abstracts away:
          - Use of FPN or not
          - Specifics of the transform method
        """
        assert method in {'RoIPoolF', 'RoIAlign'}, \
            'Unknown pooling method: {}'.format(method)

        if isinstance(blobs_in, list):
            # FPN case: add RoIFeatureTransform to each FPN level
            device_id = blobs_in[0].get_device()
            k_max = cfg.FPN.ROI_MAX_LEVEL  # coarsest level of pyramid
            k_min = cfg.FPN.ROI_MIN_LEVEL  # finest level of pyramid
            assert len(blobs_in) == k_max - k_min + 1
            bl_out_list = []
            for lvl in range(k_min, k_max + 1):
                bl_in = blobs_in[k_max - lvl]  # blobs_in is in reversed order
                sc = spatial_scale[k_max - lvl]  # in reversed order
                bl_rois = blob_rois + '_fpn' + str(lvl)
                if len(rpn_ret[bl_rois]):
                    rois = Variable(torch.from_numpy(rpn_ret[bl_rois])).cuda(device_id)
                    if method == 'RoIPoolF':
                        # Warning!: Not check if implementation matches Detectron
                        xform_out = ROIPool((resolution, resolution), sc)(bl_in, rois)
                    elif method == 'RoIAlign':
                        xform_out = ROIAlign(
                            (resolution, resolution), sc, sampling_ratio)(bl_in, rois)
                    bl_out_list.append(xform_out)

            # The pooled features from all levels are concatenated along the
            # batch dimension into a single 4D tensor.
            xform_shuffled = torch.cat(bl_out_list, dim=0)

            # Unshuffle to match rois from dataloader
            device_id = xform_shuffled.get_device()
            restore_bl = rpn_ret[blob_rois + '_idx_restore_int32']
            restore_bl = Variable(
                torch.from_numpy(restore_bl.astype('int64', copy=False))).cuda(device_id)
            xform_out = xform_shuffled[restore_bl]
        else:
            # Single feature level
            # rois: holds R regions of interest, each is a 5-tuple
            # (batch_idx, x1, y1, x2, y2) specifying an image batch index and a
            # rectangle (x1, y1, x2, y2)
            device_id = blobs_in.get_device()
            rois = Variable(torch.from_numpy(rpn_ret[blob_rois])).cuda(device_id)
            if method == 'RoIPoolF':
                xform_out = RoIPool((resolution, resolution), spatial_scale)(blobs_in, rois)
            elif method == 'RoIAlign':
                xform_out = ROIAlign(
                    (resolution, resolution), spatial_scale, sampling_ratio)(blobs_in, rois)

        return xform_out

    @check_inference
    def convbody_net(self, data):
        """For inference. Run Conv Body only"""
        blob_conv = self.Conv_Body(data)
        if cfg.FPN.FPN_ON:
            # Retain only the blobs that will be used for RoI heads. `blob_conv` may include
            # extra blobs that are used for RPN proposals, but not for RoI heads.
            blob_conv = blob_conv[-self.num_roi_levels:]
        return blob_conv

    @property
    def detectron_weight_mapping(self):
        if self.mapping_to_detectron is None:
            d_wmap = {}  # detectron_weight_mapping
            d_orphan = []  # detectron orphan weight list
            for name, m_child in self.named_children():
                if list(m_child.parameters()):  # if module has any parameter
                    child_map, child_orphan = m_child.detectron_weight_mapping()
                    d_orphan.extend(child_orphan)
                    for key, value in child_map.items():
                        new_key = name + '.' + key
                        d_wmap[new_key] = value
            self.mapping_to_detectron = d_wmap
            self.orphans_in_detectron = d_orphan

        return self.mapping_to_detectron, self.orphans_in_detectron

    def _add_loss(self, return_dict, key, value):
        """Add loss tensor to returned dictionary"""
        return_dict['losses'][key] = value
