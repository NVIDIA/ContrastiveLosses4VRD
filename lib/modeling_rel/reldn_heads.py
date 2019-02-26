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
from modeling_rel.sparse_targets_rel import FrequencyBias

logger = logging.getLogger(__name__)


class reldn_head(nn.Module):
    def __init__(self, dim_in):
        super().__init__()
        dim_in_final = dim_in // 3
        self.dim_in_final = dim_in_final
            
        if cfg.MODEL.USE_BG:
            num_prd_classes = cfg.MODEL.NUM_PRD_CLASSES + 1
        else:
            num_prd_classes = cfg.MODEL.NUM_PRD_CLASSES
            
        if cfg.MODEL.RUN_BASELINE:
            # only run it on testing mode
            self.freq_bias = FrequencyBias(cfg.TEST.DATASETS[0])
            return
        
        self.prd_cls_feats = nn.Sequential(
            nn.Linear(dim_in, dim_in // 2),
            nn.LeakyReLU(0.1),
            nn.Linear(dim_in // 2, dim_in_final),
            nn.LeakyReLU(0.1))
        self.prd_cls_scores = nn.Linear(dim_in_final, num_prd_classes)
        
        if cfg.MODEL.USE_FREQ_BIAS:
            # Assume we are training/testing on only one dataset
            if len(cfg.TRAIN.DATASETS):
                self.freq_bias = FrequencyBias(cfg.TRAIN.DATASETS[0])
            else:
                self.freq_bias = FrequencyBias(cfg.TEST.DATASETS[0])

        if cfg.MODEL.USE_SPATIAL_FEAT:
            self.spt_cls_feats = nn.Sequential(
                nn.Linear(28, 64),
                nn.LeakyReLU(0.1),
                nn.Linear(64, 64),
                nn.LeakyReLU(0.1))
            self.spt_cls_scores = nn.Linear(64, num_prd_classes)
        
        if cfg.MODEL.ADD_SO_SCORES:
            self.prd_sbj_scores = nn.Linear(dim_in_final, num_prd_classes)
            self.prd_obj_scores = nn.Linear(dim_in_final, num_prd_classes)
        
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

    # spo_feat will be concatenation of SPO
    def forward(self, spo_feat, spt_feat=None, sbj_labels=None, obj_labels=None, sbj_feat=None, obj_feat=None):

        device_id = spo_feat.get_device()
        if sbj_labels is not None:
            sbj_labels = Variable(torch.from_numpy(sbj_labels.astype('int64'))).cuda(device_id)
        if obj_labels is not None:
            obj_labels = Variable(torch.from_numpy(obj_labels.astype('int64'))).cuda(device_id)

        if cfg.MODEL.RUN_BASELINE:
            assert sbj_labels is not None and obj_labels is not None
            prd_cls_scores = self.freq_bias.rel_index_with_labels(torch.stack((sbj_labels, obj_labels), 1))
            prd_cls_scores = F.softmax(prd_cls_scores, dim=1)
            return prd_cls_scores, prd_cls_scores, None, prd_cls_scores, None, None
        
        if spo_feat.dim() == 4:
            spo_feat = spo_feat.squeeze(3).squeeze(2)
        prd_cls_feats = self.prd_cls_feats(spo_feat)
        prd_vis_scores = self.prd_cls_scores(prd_cls_feats)
        sbj_cls_scores = None
        obj_cls_scores = None
            
        if cfg.MODEL.USE_FREQ_BIAS:
            assert sbj_labels is not None and obj_labels is not None
            prd_bias_scores = self.freq_bias.rel_index_with_labels(torch.stack((sbj_labels, obj_labels), 1))
        
        if cfg.MODEL.USE_SPATIAL_FEAT:
            assert spt_feat is not None
            device_id = spo_feat.get_device()
            spt_feat = Variable(torch.from_numpy(spt_feat.astype('float32'))).cuda(device_id)
            spt_cls_feats = self.spt_cls_feats(spt_feat)
            prd_spt_scores = self.spt_cls_scores(spt_cls_feats)
        else:
            prd_spt_scores = None
            
        if cfg.MODEL.ADD_SO_SCORES:
            prd_sbj_scores = self.prd_sbj_scores(sbj_feat)
            prd_obj_scores = self.prd_obj_scores(obj_feat)
            
        if cfg.MODEL.ADD_SCORES_ALL:
            ttl_cls_scores = torch.tensor(prd_vis_scores)
            if cfg.MODEL.USE_FREQ_BIAS:
                ttl_cls_scores += prd_bias_scores
            if cfg.MODEL.USE_SPATIAL_FEAT:
                ttl_cls_scores += prd_spt_scores
            if cfg.MODEL.ADD_SO_SCORES:
                ttl_cls_scores += prd_sbj_scores + prd_obj_scores
        else:
            ttl_cls_scores = None
            
        if not self.training:
            prd_vis_scores = F.softmax(prd_vis_scores, dim=1)
            if cfg.MODEL.USE_FREQ_BIAS:
                prd_bias_scores = F.softmax(prd_bias_scores, dim=1)
            if cfg.MODEL.USE_SPATIAL_FEAT:
                prd_spt_scores = F.softmax(prd_spt_scores, dim=1)
            if cfg.MODEL.ADD_SCORES_ALL:
                ttl_cls_scores = F.softmax(ttl_cls_scores, dim=1)
        
        return prd_vis_scores, prd_bias_scores, prd_spt_scores, ttl_cls_scores, sbj_cls_scores, obj_cls_scores


def reldn_losses(prd_cls_scores, prd_labels_int32, fg_only=False):
    device_id = prd_cls_scores.get_device()
    prd_labels = Variable(torch.from_numpy(prd_labels_int32.astype('int64'))).cuda(device_id)
    loss_cls_prd = F.cross_entropy(prd_cls_scores, prd_labels)
    # class accuracy
    prd_cls_preds = prd_cls_scores.max(dim=1)[1].type_as(prd_labels)
    accuracy_cls_prd = prd_cls_preds.eq(prd_labels).float().mean(dim=0)

    return loss_cls_prd, accuracy_cls_prd


def reldn_contrastive_losses(prd_scores_sbj_pos, prd_scores_obj_pos, rel_ret):
    # sbj
    prd_probs_sbj_pos = F.softmax(prd_scores_sbj_pos, dim=1)
    sbj_pair_pos_batch, sbj_pair_neg_batch, sbj_target = split_pos_neg_spo_agnostic(
        prd_probs_sbj_pos, rel_ret['binary_labels_sbj_pos_int32'], rel_ret['inds_unique_sbj_pos'], rel_ret['inds_reverse_sbj_pos'])
    sbj_contrastive_loss = F.margin_ranking_loss(sbj_pair_pos_batch, sbj_pair_neg_batch, sbj_target, margin=cfg.MODEL.NODE_CONTRASTIVE_MARGIN)
    # obj
    prd_probs_obj_pos = F.softmax(prd_scores_obj_pos, dim=1)
    obj_pair_pos_batch, obj_pair_neg_batch, obj_target = split_pos_neg_spo_agnostic(
        prd_probs_obj_pos, rel_ret['binary_labels_obj_pos_int32'], rel_ret['inds_unique_obj_pos'], rel_ret['inds_reverse_obj_pos'])
    obj_contrastive_loss = F.margin_ranking_loss(obj_pair_pos_batch, obj_pair_neg_batch, obj_target, margin=cfg.MODEL.NODE_CONTRASTIVE_MARGIN)
    
    return sbj_contrastive_loss, obj_contrastive_loss


def reldn_so_contrastive_losses(prd_scores_sbj_pos, prd_scores_obj_pos, rel_ret):
    # sbj
    prd_probs_sbj_pos = F.softmax(prd_scores_sbj_pos, dim=1)
    sbj_pair_pos_batch, sbj_pair_neg_batch, sbj_target = split_pos_neg_so_aware(
        prd_probs_sbj_pos,
        rel_ret['binary_labels_sbj_pos_int32'], rel_ret['inds_unique_sbj_pos'], rel_ret['inds_reverse_sbj_pos'],
        rel_ret['sbj_labels_sbj_pos_int32'], rel_ret['obj_labels_sbj_pos_int32'], 's')
    sbj_so_contrastive_loss = F.margin_ranking_loss(sbj_pair_pos_batch, sbj_pair_neg_batch, sbj_target, margin=cfg.MODEL.NODE_CONTRASTIVE_SO_AWARE_MARGIN)
    # obj
    prd_probs_obj_pos = F.softmax(prd_scores_obj_pos, dim=1)
    obj_pair_pos_batch, obj_pair_neg_batch, obj_target = split_pos_neg_so_aware(
        prd_probs_obj_pos,
        rel_ret['binary_labels_obj_pos_int32'], rel_ret['inds_unique_obj_pos'], rel_ret['inds_reverse_obj_pos'],
        rel_ret['sbj_labels_obj_pos_int32'], rel_ret['obj_labels_obj_pos_int32'], 'o')
    obj_so_contrastive_loss = F.margin_ranking_loss(obj_pair_pos_batch, obj_pair_neg_batch, obj_target, margin=cfg.MODEL.NODE_CONTRASTIVE_SO_AWARE_MARGIN)
    
    return sbj_so_contrastive_loss, obj_so_contrastive_loss


def reldn_p_contrastive_losses(prd_scores_sbj_pos, prd_scores_obj_pos, prd_bias_scores_sbj_pos, prd_bias_scores_obj_pos, rel_ret):
    # sbj
    prd_probs_sbj_pos = F.softmax(prd_scores_sbj_pos, dim=1)
    prd_bias_probs_sbj_pos = F.softmax(prd_bias_scores_sbj_pos, dim=1)
    sbj_pair_pos_batch, sbj_pair_neg_batch, sbj_target = split_pos_neg_p_aware(
        prd_probs_sbj_pos,
        prd_bias_probs_sbj_pos,
        rel_ret['binary_labels_sbj_pos_int32'], rel_ret['inds_unique_sbj_pos'], rel_ret['inds_reverse_sbj_pos'],
        rel_ret['prd_labels_sbj_pos_int32'])
    sbj_p_contrastive_loss = F.margin_ranking_loss(sbj_pair_pos_batch, sbj_pair_neg_batch, sbj_target, margin=cfg.MODEL.NODE_CONTRASTIVE_P_AWARE_MARGIN)
    # obj
    prd_probs_obj_pos = F.softmax(prd_scores_obj_pos, dim=1)
    prd_bias_probs_obj_pos = F.softmax(prd_bias_scores_obj_pos, dim=1)
    obj_pair_pos_batch, obj_pair_neg_batch, obj_target = split_pos_neg_p_aware(
        prd_probs_obj_pos,
        prd_bias_probs_obj_pos,
        rel_ret['binary_labels_obj_pos_int32'], rel_ret['inds_unique_obj_pos'], rel_ret['inds_reverse_obj_pos'],
        rel_ret['prd_labels_obj_pos_int32'])
    obj_p_contrastive_loss = F.margin_ranking_loss(obj_pair_pos_batch, obj_pair_neg_batch, obj_target, margin=cfg.MODEL.NODE_CONTRASTIVE_P_AWARE_MARGIN)
    
    return sbj_p_contrastive_loss, obj_p_contrastive_loss


def split_pos_neg_spo_agnostic(prd_probs, binary_labels_pos, inds_unique_pos, inds_reverse_pos):
    device_id = prd_probs.get_device()
    prd_pos_probs = 1 - prd_probs[:, 0]  # shape is (#rels,)
    # loop over each group
    pair_pos_batch = torch.ones(1).cuda(device_id)  # a dummy sample in the batch in case there is no real sample
    pair_neg_batch = torch.zeros(1).cuda(device_id)  # a dummy sample in the batch in case there is no real sample
    for i in range(inds_unique_pos.shape[0]):
        inds = np.where(inds_reverse_pos == i)[0]
        prd_pos_probs_i = prd_pos_probs[inds]
        binary_labels_pos_i = binary_labels_pos[inds]
        pair_pos_inds = np.where(binary_labels_pos_i > 0)[0]
        pair_neg_inds = np.where(binary_labels_pos_i == 0)[0]
        if pair_pos_inds.size == 0 or pair_neg_inds.size == 0:  # ignore this node if either pos or neg does not exist
            continue
        prd_pos_probs_i_pair_pos = prd_pos_probs_i[pair_pos_inds]
        prd_pos_probs_i_pair_neg = prd_pos_probs_i[pair_neg_inds]
        min_prd_pos_probs_i_pair_pos = torch.min(prd_pos_probs_i_pair_pos)
        max_prd_pos_probs_i_pair_neg = torch.max(prd_pos_probs_i_pair_neg)
        pair_pos_batch = torch.cat((pair_pos_batch, min_prd_pos_probs_i_pair_pos.unsqueeze(0)))
        pair_neg_batch = torch.cat((pair_neg_batch, max_prd_pos_probs_i_pair_neg.unsqueeze(0)))

    target = torch.ones_like(pair_pos_batch).cuda(device_id)
        
    return pair_pos_batch, pair_neg_batch, target


def split_pos_neg_so_aware(prd_probs, binary_labels_pos, inds_unique_pos, inds_reverse_pos, sbj_labels_pos, obj_labels_pos, s_or_o):
    device_id = prd_probs.get_device()
    prd_pos_probs = 1 - prd_probs[:, 0]  # shape is (#rels,)
    # loop over each group
    pair_pos_batch = torch.ones(1).cuda(device_id)  # a dummy sample in the batch in case there is no real sample
    pair_neg_batch = torch.zeros(1).cuda(device_id)  # a dummy sample in the batch in case there is no real sample
    for i in range(inds_unique_pos.shape[0]):
        inds = np.where(inds_reverse_pos == i)[0]
        prd_pos_probs_i = prd_pos_probs[inds]
        binary_labels_pos_i = binary_labels_pos[inds]
        sbj_labels_pos_i = sbj_labels_pos[inds]
        obj_labels_pos_i = obj_labels_pos[inds]
        pair_pos_inds = np.where(binary_labels_pos_i > 0)[0]
        pair_neg_inds = np.where(binary_labels_pos_i == 0)[0]
        if pair_pos_inds.size == 0 or pair_neg_inds.size == 0:  # ignore this node if either pos or neg does not exist
            continue
        prd_pos_probs_i_pair_pos = prd_pos_probs_i[pair_pos_inds]
        prd_pos_probs_i_pair_neg = prd_pos_probs_i[pair_neg_inds]
        sbj_labels_i_pair_pos = sbj_labels_pos_i[pair_pos_inds]
        obj_labels_i_pair_pos = obj_labels_pos_i[pair_pos_inds]
        sbj_labels_i_pair_neg = sbj_labels_pos_i[pair_neg_inds]
        obj_labels_i_pair_neg = obj_labels_pos_i[pair_neg_inds]
        max_prd_pos_probs_i_pair_neg = torch.max(prd_pos_probs_i_pair_neg)  # this is fixed for a given i
        if s_or_o == 's':
            # get all unique object labels
            unique_obj_labels, inds_unique_obj_labels, inds_reverse_obj_labels = np.unique(
                obj_labels_i_pair_pos, return_index=True, return_inverse=True, axis=0)
            for j in range(inds_unique_obj_labels.shape[0]):
                # get min pos
                inds_j = np.where(inds_reverse_obj_labels == j)[0]
                prd_pos_probs_i_pos_j = prd_pos_probs_i_pair_pos[inds_j]
                min_prd_pos_probs_i_pos_j = torch.min(prd_pos_probs_i_pos_j)
                # get max neg
                neg_j_inds = np.where(obj_labels_i_pair_neg == unique_obj_labels[j])[0]
                if neg_j_inds.size == 0:
                    if cfg.MODEL.USE_SPO_AGNOSTIC_COMPENSATION:
                        pair_pos_batch = torch.cat((pair_pos_batch, min_prd_pos_probs_i_pos_j.unsqueeze(0)))
                        pair_neg_batch = torch.cat((pair_neg_batch, max_prd_pos_probs_i_pair_neg.unsqueeze(0)))
                    continue
                prd_pos_probs_i_neg_j = prd_pos_probs_i_pair_neg[neg_j_inds]
                max_prd_pos_probs_i_neg_j = torch.max(prd_pos_probs_i_neg_j)
                pair_pos_batch = torch.cat((pair_pos_batch, min_prd_pos_probs_i_pos_j.unsqueeze(0)))
                pair_neg_batch = torch.cat((pair_neg_batch, max_prd_pos_probs_i_neg_j.unsqueeze(0)))
        else:
            # get all unique subject labels
            unique_sbj_labels, inds_unique_sbj_labels, inds_reverse_sbj_labels = np.unique(
                sbj_labels_i_pair_pos, return_index=True, return_inverse=True, axis=0)
            for j in range(inds_unique_sbj_labels.shape[0]):
                # get min pos
                inds_j = np.where(inds_reverse_sbj_labels == j)[0]
                prd_pos_probs_i_pos_j = prd_pos_probs_i_pair_pos[inds_j]
                min_prd_pos_probs_i_pos_j = torch.min(prd_pos_probs_i_pos_j)
                # get max neg
                neg_j_inds = np.where(sbj_labels_i_pair_neg == unique_sbj_labels[j])[0]
                if neg_j_inds.size == 0:
                    if cfg.MODEL.USE_SPO_AGNOSTIC_COMPENSATION:
                        pair_pos_batch = torch.cat((pair_pos_batch, min_prd_pos_probs_i_pos_j.unsqueeze(0)))
                        pair_neg_batch = torch.cat((pair_neg_batch, max_prd_pos_probs_i_pair_neg.unsqueeze(0)))
                    continue
                prd_pos_probs_i_neg_j = prd_pos_probs_i_pair_neg[neg_j_inds]
                max_prd_pos_probs_i_neg_j = torch.max(prd_pos_probs_i_neg_j)
                pair_pos_batch = torch.cat((pair_pos_batch, min_prd_pos_probs_i_pos_j.unsqueeze(0)))
                pair_neg_batch = torch.cat((pair_neg_batch, max_prd_pos_probs_i_neg_j.unsqueeze(0)))

    target = torch.ones_like(pair_pos_batch).cuda(device_id)

    return pair_pos_batch, pair_neg_batch, target


def split_pos_neg_p_aware(prd_probs, prd_bias_probs, binary_labels_pos, inds_unique_pos, inds_reverse_pos, prd_labels_pos):
    device_id = prd_probs.get_device()
    prd_pos_probs = 1 - prd_probs[:, 0]  # shape is (#rels,)
    prd_labels_det = prd_probs[:, 1:].argmax(dim=1).data.cpu().numpy() + 1  # prd_probs is a torch.tensor, exlucding background
    # loop over each group
    pair_pos_batch = torch.ones(1).cuda(device_id)  # a dummy sample in the batch in case there is no real sample
    pair_neg_batch = torch.zeros(1).cuda(device_id)  # a dummy sample in the batch in case there is no real sample
    for i in range(inds_unique_pos.shape[0]):
        inds = np.where(inds_reverse_pos == i)[0]
        prd_pos_probs_i = prd_pos_probs[inds]
        prd_labels_pos_i = prd_labels_pos[inds]
        prd_labels_det_i = prd_labels_det[inds]
        binary_labels_pos_i = binary_labels_pos[inds]
        pair_pos_inds = np.where(binary_labels_pos_i > 0)[0]
        pair_neg_inds = np.where(binary_labels_pos_i == 0)[0]
        if pair_pos_inds.size == 0 or pair_neg_inds.size == 0:  # ignore this node if either pos or neg does not exist
            continue
        prd_pos_probs_i_pair_pos = prd_pos_probs_i[pair_pos_inds]
        prd_pos_probs_i_pair_neg = prd_pos_probs_i[pair_neg_inds]
        prd_labels_i_pair_pos = prd_labels_pos_i[pair_pos_inds]
        prd_labels_i_pair_neg = prd_labels_det_i[pair_neg_inds]
        max_prd_pos_probs_i_pair_neg = torch.max(prd_pos_probs_i_pair_neg)  # this is fixed for a given i
        unique_prd_labels, inds_unique_prd_labels, inds_reverse_prd_labels = np.unique(
            prd_labels_i_pair_pos, return_index=True, return_inverse=True, axis=0)
        for j in range(inds_unique_prd_labels.shape[0]):
            # get min pos
            inds_j = np.where(inds_reverse_prd_labels == j)[0]
            prd_pos_probs_i_pos_j = prd_pos_probs_i_pair_pos[inds_j]
            min_prd_pos_probs_i_pos_j = torch.min(prd_pos_probs_i_pos_j)
            # get max neg
            neg_j_inds = np.where(prd_labels_i_pair_neg == unique_prd_labels[j])[0]
            if neg_j_inds.size == 0:
                if cfg.MODEL.USE_SPO_AGNOSTIC_COMPENSATION:
                    pair_pos_batch = torch.cat((pair_pos_batch, min_prd_pos_probs_i_pos_j.unsqueeze(0)))
                    pair_neg_batch = torch.cat((pair_neg_batch, max_prd_pos_probs_i_pair_neg.unsqueeze(0)))
                continue
            prd_pos_probs_i_neg_j = prd_pos_probs_i_pair_neg[neg_j_inds]
            max_prd_pos_probs_i_neg_j = torch.max(prd_pos_probs_i_neg_j)
            pair_pos_batch = torch.cat((pair_pos_batch, min_prd_pos_probs_i_pos_j.unsqueeze(0)))
            pair_neg_batch = torch.cat((pair_neg_batch, max_prd_pos_probs_i_neg_j.unsqueeze(0)))

    target = torch.ones_like(pair_pos_batch).cuda(device_id)
        
    return pair_pos_batch, pair_neg_batch, target
