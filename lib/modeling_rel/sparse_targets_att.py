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

from core.config import cfg
from modeling_rel.get_dataset_counts_att import get_att_counts


logger = logging.getLogger(__name__)


# This module is adapted from Rowan Zellers:
# https://github.com/rowanz/neural-motifs/blob/master/lib/sparse_targets.py
# Modified for this project
class FrequencyBias(nn.Module):
    """
    The goal of this is to provide a simplified way of computing
    P(predicate | obj1, obj2, img).
    """

    def __init__(self, eps=1e-3):
        super(FrequencyBias, self).__init__()

        att_fg_matrix, att_bg_matrix = get_att_counts()
        att_bg_matrix += 1
        att_fg_matrix[:, 0] = att_bg_matrix

        att_pred_dist = np.log(att_fg_matrix / (att_fg_matrix.sum(1)[:, None] + 1e-08) + eps)

        self.num_att_objs = att_pred_dist.shape[0]
        att_pred_dist = torch.FloatTensor(att_pred_dist).view(-1, att_pred_dist.shape[1])

        self.att_baseline = nn.Embedding(att_pred_dist.size(0), att_pred_dist.size(1))
        self.att_baseline.weight.data = att_pred_dist
        
        logger.info('Frequency bias tables loaded.')

    def att_index_with_labels(self, labels):
        """
        :param labels: [batch_size, 2] 
        :return: 
        """
        return self.att_baseline(labels)
