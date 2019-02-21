"""
Written by Ji Zhang, 2019
Some functions are adapted from Rowan Zellers
Original source:
https://github.com/rowanz/neural-motifs/blob/master/lib/evaluation/sg_eval.py
"""

import os
import numpy as np
import logging
from six.moves import cPickle as pickle
from tqdm import tqdm

from core.config import cfg
from functools import reduce
from utils.boxes import bbox_overlaps
from datasets_rel.ap_eval_rel import ap_eval, prepare_mAP_dets

from .pytorch_misc import intersect_2d, argsort_desc

np.set_printoptions(precision=3)

logger = logging.getLogger(__name__)

att_k = 2
topk = 100


def eval_att_results(all_results, output_dir, do_val):
        
    if cfg.TEST.DATASETS[0].find('oi') >= 0:
        eval_ap = True
    else:
        eval_ap = False

    recalls = {1: 0, 5: 0, 10: 0, 20: 0, 50: 0, 100: 0}
    if do_val:
        all_gt_cnt = 0
    
    topk_dets = []
    topk_dets_for_ap = []
    for res in tqdm(all_results):
        
        # in oi_all_att some images have no dets
        if res['att_scores'] is None:
            det_boxes_o_top = np.zeros((0, 4), dtype=np.float32)
            det_labels_o_top = np.zeros(0, dtype=np.int32)
            det_labels_a_top = np.zeros(0, dtype=np.int32)
            det_scores_top = np.zeros(0, dtype=np.float32)
        else:
            det_boxes_obj = res['obj_boxes']  # (#num_att, 4)
            det_labels_obj = res['obj_labels']  # (#num_att,)
            det_scores_obj = res['obj_scores']  # (#num_att,)
            det_scores_att = res['att_scores'][:, 1:]  # (#num_att, #num_fg_atts)
            det_scores_pos = 1 - res['att_scores'][:, 0]

            det_labels_att = np.argsort(-det_scores_att, axis=1)
            det_scores_att = -np.sort(-det_scores_att, axis=1)

            det_scores_so = det_scores_obj * det_scores_obj
            det_scores_spo = det_scores_so[:, None] * det_scores_att[:, :att_k]
            det_scores_inds = argsort_desc(det_scores_spo)[:topk]
            det_scores_top = det_scores_spo[det_scores_inds[:, 0], det_scores_inds[:, 1]]
            det_boxes_so_top = np.hstack(
                (det_boxes_obj[det_scores_inds[:, 0]], det_boxes_obj[det_scores_inds[:, 0]]))
            det_labels_p_top = det_labels_att[det_scores_inds[:, 0], det_scores_inds[:, 1]]
            det_labels_spo_top = np.vstack(
                (det_labels_obj[det_scores_inds[:, 0]], det_labels_p_top, det_labels_obj[det_scores_inds[:, 0]])).transpose()

            det_boxes_s_top = det_boxes_so_top[:, :4]
            det_boxes_o_top = det_boxes_so_top[:, 4:]
            assert (det_boxes_s_top == det_boxes_o_top).all()
            det_labels_s_top = det_labels_spo_top[:, 0]
            det_labels_a_top = det_labels_spo_top[:, 1]
            det_labels_o_top = det_labels_spo_top[:, 2]
            assert (det_labels_s_top == det_labels_o_top).all()

        topk_dets.append(dict(image=res['image'],
                              det_boxes_o_top=det_boxes_o_top,
                              det_labels_o_top=det_labels_o_top,
                              det_labels_a_top=det_labels_a_top,
                              det_scores_top=det_scores_top))
        if eval_ap:
            topk_dets_for_ap.append(dict(image=res['image'],
                                         det_boxes_s_top=det_boxes_o_top,
                                         det_boxes_o_top=det_boxes_o_top,
                                         det_labels_s_top=det_labels_o_top,
                                         det_labels_p_top=np.zeros(det_labels_o_top.shape[0], dtype=np.int32),
                                         det_labels_o_top=det_labels_a_top,
                                         det_scores_top=det_scores_top))
        
        if do_val:
            gt_boxes_obj = res['gt_obj_boxes']  # (#num_gt, 4)
            gt_labels_obj = res['gt_obj_labels']  # (#num_gt,)
            gt_labels_att = res['gt_att_labels']  # (#num_gt,)
            gt_boxes_so = np.hstack((gt_boxes_obj, gt_boxes_obj))
            gt_labels_spo = np.vstack((gt_labels_obj, gt_labels_att, gt_labels_obj)).transpose()
            # Compute recall. It's most efficient to match once and then do recall after
            # det_boxes_so_top is (#num_att, 8)
            # det_labels_spo_top is (#num_att, 3)
            pred_to_gt = _compute_pred_matches(
                gt_labels_spo, det_labels_spo_top,
                gt_boxes_so, det_boxes_so_top)
            all_gt_cnt += gt_labels_spo.shape[0]
            for k in recalls:
                match = reduce(np.union1d, pred_to_gt[:k])
                recalls[k] += len(match)
            
            topk_dets[-1].update(dict(gt_boxes_obj=gt_boxes_obj,
                                      gt_labels_obj=gt_labels_obj,
                                      gt_labels_att=gt_labels_att))
            if eval_ap:
                topk_dets_for_ap[-1].update(dict(gt_boxes_sbj=gt_boxes_obj,
                                                 gt_boxes_obj=gt_boxes_obj,
                                                 gt_labels_sbj=gt_labels_obj,
                                                 gt_labels_obj=gt_labels_att,
                                                 gt_labels_prd=np.zeros(gt_labels_obj.shape[0], dtype=np.int32)))

    if do_val:
        for k in recalls:
            recalls[k] = float(recalls[k]) / (float(all_gt_cnt) + 1e-12)
        print_stats(recalls)
        if eval_ap:
            # prepare dets for each class
            logger.info('Preparing dets for mAP...')
            cls_image_ids, cls_dets, cls_gts, npos = prepare_mAP_dets(topk_dets_for_ap, 1)
            rel_prd_cats = ['is']
            # rel APs
            rel_mAP = 0.
            for c in range(1):
                rec, prec, ap = ap_eval(cls_image_ids[c], cls_dets[c], cls_gts[c], npos[c], True)
                rel_mAP += ap
                print('rel AP for class {}: {:.6f}'.format(rel_prd_cats[c], ap))
            rel_mAP /= 1.
            print('rel mAP: {:.6f}'.format(rel_mAP))
            # phr APs
            phr_mAP = 0.
            for c in range(1):
                rec, prec, ap = ap_eval(cls_image_ids[c], cls_dets[c], cls_gts[c], npos[c], False)
                phr_mAP += ap
                print('phr AP for class {}: {:.6f}'.format(rel_prd_cats[c], ap))
            phr_mAP /= 1.
            print('phr mAP: {:.6f}'.format(phr_mAP))
            # total: 0.4 x rel_mAP + 0.2 x R@50 + 0.4 x phr_mAP
            final_score = 0.4 * rel_mAP + 0.2 * recalls[50] + 0.4 * phr_mAP
            print('final_score: {:.6f}'.format(final_score))

    print('Saving topk dets...')
    topk_dets_f = os.path.join(output_dir, 'att_detections_topk.pkl')
    with open(topk_dets_f, 'wb') as f:
        pickle.dump(topk_dets, f, pickle.HIGHEST_PROTOCOL)
    logger.info('topk_dets size: {}'.format(len(topk_dets)))
    print('Done.')


def print_stats(recalls, ratios=None):
    print('====================== ' + 'sgdet' + ' ============================')
    for k, v in recalls.items():
        print('R@%i: %f' % (k, v))


# This function is adapted from Rowan Zellers' code:
# https://github.com/rowanz/neural-motifs/blob/master/lib/evaluation/sg_eval.py
# Modified for this project to work with PyTorch v0.4
def _compute_pred_matches(gt_triplets, pred_triplets,
                 gt_boxes, pred_boxes, iou_thresh=0.5, phrdet=False):
    """
    Given a set of predicted triplets, return the list of matching GT's for each of the
    given predictions
    :param gt_triplets: 
    :param pred_triplets: 
    :param gt_boxes: 
    :param pred_boxes: 
    :param iou_thresh: 
    :return: 
    """
    # This performs a matrix multiplication-esque thing between the two arrays
    # Instead of summing, we want the equality, so we reduce in that way
    # The rows correspond to GT triplets, columns to pred triplets
    keeps = intersect_2d(gt_triplets, pred_triplets)
    gt_has_match = keeps.any(1)
    pred_to_gt = [[] for x in range(pred_boxes.shape[0])]
    for gt_ind, gt_box, keep_inds in zip(np.where(gt_has_match)[0],
                                         gt_boxes[gt_has_match],
                                         keeps[gt_has_match],
                                         ):
        boxes = pred_boxes[keep_inds]
        if phrdet:
            # Evaluate where the union box > 0.5
            gt_box_union = gt_box.reshape((2, 4))
            gt_box_union = np.concatenate((gt_box_union.min(0)[:2], gt_box_union.max(0)[2:]), 0)

            box_union = boxes.reshape((-1, 2, 4))
            box_union = np.concatenate((box_union.min(1)[:,:2], box_union.max(1)[:,2:]), 1)

            gt_box_union = gt_box_union.astype(dtype=np.float32, copy=False)
            box_union = box_union.astype(dtype=np.float32, copy=False)
            inds = bbox_overlaps(gt_box_union[None], 
                                box_union = box_union)[0] >= iou_thresh

        else:
            gt_box = gt_box.astype(dtype=np.float32, copy=False)
            boxes = boxes.astype(dtype=np.float32, copy=False)
            sub_iou = bbox_overlaps(gt_box[None,:4], boxes[:, :4])[0]
            obj_iou = bbox_overlaps(gt_box[None,4:], boxes[:, 4:])[0]

            inds = (sub_iou >= iou_thresh) & (obj_iou >= iou_thresh)

        for i in np.where(keep_inds)[0][inds]:
            pred_to_gt[i].append(int(gt_ind))
    return pred_to_gt
