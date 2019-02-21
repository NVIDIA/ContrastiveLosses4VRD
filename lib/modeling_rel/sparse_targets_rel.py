"""
Some functions are adapted from Rowan Zellers:
https://github.com/rowanz/neural-motifs
"""
import os
import torch.nn as nn
import torch
from torch.autograd import Variable
import numpy as np
import logging
from six.moves import cPickle as pickle

from core.config import cfg
from modeling_rel.get_dataset_counts_rel import get_rel_counts


logger = logging.getLogger(__name__)


# This module is adapted from Rowan Zellers:
# https://github.com/rowanz/neural-motifs/blob/master/lib/sparse_targets.py
# Modified for this project
class FrequencyBias(nn.Module):
    """
    The goal of this is to provide a simplified way of computing
    P(predicate | obj1, obj2, img).
    """

    def __init__(self, ds_name, eps=1e-3):
        super(FrequencyBias, self).__init__()

        if ds_name.find('vg') >= 0:
            ds_name = 'vg'
        elif ds_name.find('oi') >= 0:
            ds_name = 'oi'
        elif ds_name.find('vrd') >= 0:
            ds_name = 'vrd'
        else:
            raise NotImplementedError

        if cfg.MODEL.USE_OVLP_FILTER:
            must_overlap = True
        else:
            must_overlap = False
        fg_matrix, bg_matrix = get_rel_counts(ds_name, must_overlap=must_overlap)
        bg_matrix += 1
        fg_matrix[:, :, 0] = bg_matrix

        pred_dist = np.log(fg_matrix / (fg_matrix.sum(2)[:, :, None] + 1e-08) + eps)

        self.num_objs = pred_dist.shape[0]
        pred_dist = torch.FloatTensor(pred_dist).view(-1, pred_dist.shape[2])

        self.rel_baseline = nn.Embedding(pred_dist.size(0), pred_dist.size(1))
        self.rel_baseline.weight.data = pred_dist
        
        logger.info('Frequency bias tables loaded.')

    def rel_index_with_labels(self, labels):
        """
        :param labels: [batch_size, 2] 
        :return: 
        """
        return self.rel_baseline(labels[:, 0] * self.num_objs + labels[:, 1])
