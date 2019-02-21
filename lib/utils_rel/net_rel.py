# Adapted by Ji Zhang in 2019
#
# Based on Detectron.pytorch/lib/utils/net.py written by Roy Tseng

import logging
import os
import numpy as np

import torch
import torch.nn.functional as F
from torch.autograd import Variable

from core.config import cfg
from utils.net import _get_lr_change_ratio
from utils.net import _CorrectMomentum

logger = logging.getLogger(__name__)


def update_learning_rate_att(optimizer, cur_lr, new_lr):
    """Update learning rate"""
    if cur_lr != new_lr:
        ratio = _get_lr_change_ratio(cur_lr, new_lr)
        if ratio > cfg.SOLVER.LOG_LR_CHANGE_THRESHOLD:
            logger.info('Changing learning rate %.6f -> %.6f', cur_lr, new_lr)
        # Update learning rate, note that different parameter may have different learning rate
        param_keys = []
        for ind, param_group in enumerate(optimizer.param_groups):
            if (ind == 1 or ind == 3) and cfg.SOLVER.BIAS_DOUBLE_LR:  # bias params
                param_group['lr'] = new_lr * 2
            else:
                param_group['lr'] = new_lr
            if ind <= 1:  # backbone params
                param_group['lr'] = cfg.SOLVER.BACKBONE_LR_SCALAR * param_group['lr']  # 0.1 * param_group['lr']
            param_keys += param_group['params']
        if cfg.SOLVER.TYPE in ['SGD'] and cfg.SOLVER.SCALE_MOMENTUM and cur_lr > 1e-7 and \
                ratio > cfg.SOLVER.SCALE_MOMENTUM_THRESHOLD:
            _CorrectMomentum(optimizer, param_keys, new_lr / cur_lr)
            

def update_learning_rate_rel(optimizer, cur_lr, new_lr):
    """Update learning rate"""
    if cur_lr != new_lr:
        ratio = _get_lr_change_ratio(cur_lr, new_lr)
        if ratio > cfg.SOLVER.LOG_LR_CHANGE_THRESHOLD:
            logger.info('Changing learning rate %.6f -> %.6f', cur_lr, new_lr)
        # Update learning rate, note that different parameter may have different learning rate
        param_keys = []
        for ind, param_group in enumerate(optimizer.param_groups):
            if (ind == 1 or ind == 3) and cfg.SOLVER.BIAS_DOUBLE_LR:  # bias params
                param_group['lr'] = new_lr * 2
            else:
                param_group['lr'] = new_lr
            if ind <= 1:  # backbone params
                param_group['lr'] = cfg.SOLVER.BACKBONE_LR_SCALAR * param_group['lr']  # 0.1 * param_group['lr']
            param_keys += param_group['params']
        if cfg.SOLVER.TYPE in ['SGD'] and cfg.SOLVER.SCALE_MOMENTUM and cur_lr > 1e-7 and \
                ratio > cfg.SOLVER.SCALE_MOMENTUM_THRESHOLD:
            _CorrectMomentum(optimizer, param_keys, new_lr / cur_lr)


def load_ckpt_rel(model, ckpt):
    """Load checkpoint"""
    
    model.load_state_dict(ckpt, strict=False)
