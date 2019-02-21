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
import json
import csv
from tqdm import tqdm

from core.config import cfg
from functools import reduce
from utils.boxes import bbox_overlaps
from datasets_rel.ap_eval_rel import ap_eval, prepare_mAP_dets

from .pytorch_misc import intersect_2d, argsort_desc

np.set_printoptions(precision=3)

logger = logging.getLogger(__name__)

prd_k = 2
att_k = 2
topk = 100

att_scalar = 1.0


def eval_rel_results(rel_results, att_results, output_dir, do_val):
        
    if cfg.TEST.DATASETS[0].find('oi') >= 0:
        eval_ap = True
    else:
        eval_ap = False
    
    recalls = {1: 0, 5: 0, 10: 0, 20: 0, 50: 0, 100: 0}
    if do_val:
        all_gt_cnt = 0
    
    topk_dets = []
    for im_i, rel_res in enumerate(tqdm(rel_results)):
        
        att_res = att_results[im_i]
        assert rel_res['image'].split('/')[-1] == att_res['image'].split('/')[-1]
        
        # in oi_all_rel some images have no dets or gts
        if rel_res['prd_scores'] is None:  # or len(res['gt_prd_labels']) == 0:
            rel_boxes_sbj = np.zeros((0, 4), dtype=np.float32)
            rel_boxes_obj = np.zeros((0, 4), dtype=np.float32)
            rel_labels_sbj = np.zeros(0, dtype=np.int32)
            rel_labels_obj = np.zeros(0, dtype=np.int32)
            rel_labels_prd = np.zeros(0, dtype=np.int32)
            rel_scores_sbj = np.zeros(0, dtype=np.float32)
            rel_scores_obj = np.zeros(0, dtype=np.float32)
            rel_scores_prd = np.zeros(0, dtype=np.float32)
        else:
            rel_boxes_sbj = rel_res['sbj_boxes']  # (#num_rel, 4)
            rel_boxes_obj = rel_res['obj_boxes']  # (#num_rel, 4)
            rel_labels_sbj = rel_res['sbj_labels']  # (#num_rel,)
            rel_labels_obj = rel_res['obj_labels']  # (#num_rel,)
            rel_scores_sbj = rel_res['sbj_scores']  # (#num_rel,)
            rel_scores_obj = rel_res['obj_scores']  # (#num_rel,)
            
            if cfg.MODEL.RUN_BASELINE:
                rel_scores_prd = rel_res['prd_scores'][:, 1:]
            else:
                rel_scores_prd = rel_res['prd_scores_ttl'][:, 1:]
            rel_labels_prd = np.argsort(-rel_scores_prd, axis=1)
            rel_scores_prd = -np.sort(-rel_scores_prd, axis=1)
            # reshape results for multiple prd
            rel_boxes_sbj = np.repeat(rel_boxes_sbj, prd_k, axis=0)
            rel_boxes_obj = np.repeat(rel_boxes_obj, prd_k, axis=0)
            rel_labels_sbj = np.repeat(rel_labels_sbj, prd_k)
            rel_labels_obj = np.repeat(rel_labels_obj, prd_k)
            rel_scores_sbj = np.repeat(rel_scores_sbj, prd_k)
            rel_scores_obj = np.repeat(rel_scores_obj, prd_k)
            rel_labels_prd = rel_labels_prd[:, :prd_k].reshape(-1)
            rel_scores_prd = rel_scores_prd[:, :prd_k].reshape(-1)
            
        if att_res['att_scores'] is None:
            att_boxes_obj = np.zeros((0, 4), dtype=np.float32)
            att_labels_obj = np.zeros(0, dtype=np.int32)
            att_scores_obj = np.zeros(0, dtype=np.float32)
            att_labels_att = np.zeros(0, dtype=np.int32)
            att_scores_att = np.zeros(0, dtype=np.float32)
        else:
            att_boxes_obj = att_res['obj_boxes']  # (#num_att, 4)
            att_labels_obj = att_res['obj_labels']  # (#num_att,)
            att_scores_obj = att_res['obj_scores']  # (#num_att,)
            att_scores_att = att_res['att_scores'][:, 1:]  # (#num_att, #num_fg_atts)
            att_labels_att = np.argsort(-att_scores_att, axis=1)
            att_scores_att = -np.sort(-att_scores_att, axis=1)
            # reshape results for multiple att
            att_boxes_obj = np.repeat(att_boxes_obj, att_k, axis=0)
            att_labels_obj = np.repeat(att_labels_obj, att_k)
            att_scores_obj = np.repeat(att_scores_obj, att_k)
            att_labels_att = att_labels_att[:, :att_k].reshape(-1)
            att_scores_att = att_scores_att[:, :att_k].reshape(-1) * att_scalar
            
        # merge rel and att
        det_boxes_sbj = np.concatenate((rel_boxes_sbj, att_boxes_obj), 0)
        det_boxes_obj = np.concatenate((rel_boxes_obj, att_boxes_obj), 0)
        det_labels_sbj = np.concatenate((rel_labels_sbj, att_labels_obj))
        det_labels_obj = np.concatenate((rel_labels_obj, att_labels_att))
        det_scores_sbj = np.concatenate((rel_scores_sbj, att_scores_obj))
        det_scores_obj = np.concatenate((rel_scores_obj, att_scores_att))
        det_labels_prd = np.concatenate((rel_labels_prd, 9 * np.ones(att_labels_obj.shape[0], dtype=np.int32)))
        det_scores_prd = np.concatenate((rel_scores_prd, np.ones(att_labels_obj.shape[0], dtype=np.float32)))
            
        det_scores_spo = det_scores_sbj * det_scores_obj * det_scores_prd
        # filter out bad relationships
        det_scores_inds = np.argsort(-det_scores_spo)[:topk]
        det_scores_top = det_scores_spo[det_scores_inds]
        valid_inds = np.where(det_scores_top > cfg.TEST.SPO_SCORE_THRESH)[0]
        det_scores_top = det_scores_top[valid_inds]
        det_scores_inds = det_scores_inds[valid_inds]
        det_boxes_s_top = det_boxes_sbj[det_scores_inds]
        det_boxes_o_top = det_boxes_obj[det_scores_inds]
        det_labels_s_top = det_labels_sbj[det_scores_inds]
        det_labels_o_top = det_labels_obj[det_scores_inds]
        det_labels_p_top = det_labels_prd[det_scores_inds]

        det_boxes_so_top = np.hstack((det_boxes_s_top, det_boxes_o_top))
        det_labels_spo_top = np.vstack((det_labels_s_top, det_labels_p_top, det_labels_o_top)).transpose()
        
        topk_dets.append(dict(image=rel_res['image'],
                              det_boxes_s_top=det_boxes_s_top,
                              det_boxes_o_top=det_boxes_o_top,
                              det_labels_s_top=det_labels_s_top,
                              det_labels_p_top=det_labels_p_top,
                              det_labels_o_top=det_labels_o_top,
                              det_scores_top=det_scores_top))
        
        if do_val:
            rel_gt_boxes_sbj = rel_res['gt_sbj_boxes']  # (#num_gt, 4)
            rel_gt_boxes_obj = rel_res['gt_obj_boxes']  # (#num_gt, 4)
            rel_gt_labels_sbj = rel_res['gt_sbj_labels']  # (#num_gt,)
            rel_gt_labels_obj = rel_res['gt_obj_labels']  # (#num_gt,)
            rel_gt_labels_prd = rel_res['gt_prd_labels']  # (#num_gt,)
            att_gt_boxes_obj = att_res['gt_obj_boxes']  # (#num_gt, 4)
            att_gt_labels_obj = att_res['gt_obj_labels']  # (#num_gt,)
            att_gt_labels_att = att_res['gt_att_labels']  # (#num_gt,)
            gt_boxes_sbj = np.concatenate((rel_gt_boxes_sbj, att_gt_boxes_obj), 0)
            gt_boxes_obj = np.concatenate((rel_gt_boxes_obj, att_gt_boxes_obj), 0)
            gt_labels_sbj = np.concatenate((rel_gt_labels_sbj, att_gt_labels_obj))
            gt_labels_obj = np.concatenate((rel_gt_labels_obj, att_gt_labels_att))
            gt_labels_prd = np.concatenate((rel_gt_labels_prd, 9 * np.ones(att_gt_labels_obj.shape[0], dtype=np.int32)))

            gt_boxes_so = np.hstack((gt_boxes_sbj, gt_boxes_obj))
            gt_labels_spo = np.vstack((gt_labels_sbj, gt_labels_prd, gt_labels_obj)).transpose()

            # Compute recall. It's most efficient to match once and then do recall after
            # det_boxes_so_top is (#num_rel, 8)
            # det_labels_spo_top is (#num_rel, 3)
            pred_to_gt = _compute_pred_matches(
                gt_labels_spo, det_labels_spo_top,
                gt_boxes_so, det_boxes_so_top)

            all_gt_cnt += gt_labels_spo.shape[0]
            for k in recalls:
                if len(pred_to_gt):
                    match = reduce(np.union1d, pred_to_gt[:k])
                else:
                    match = []
                recalls[k] += len(match)

            topk_dets[-1].update(dict(gt_boxes_sbj=gt_boxes_sbj,
                                      gt_boxes_obj=gt_boxes_obj,
                                      gt_labels_sbj=gt_labels_sbj,
                                      gt_labels_prd=gt_labels_prd,
                                      gt_labels_obj=gt_labels_obj))
    
    if do_val:
        for k in recalls:
            recalls[k] = float(recalls[k]) / (float(all_gt_cnt) + 1e-12)
        print_stats(recalls)
        if eval_ap:
            # prepare dets for each class
            cls_num = 10
            logger.info('Preparing dets for mAP...')
            cls_image_ids, cls_dets, cls_gts, npos = prepare_mAP_dets(topk_dets, cls_num)
            with open(cfg.DATA_DIR + '/openimages_v4/rel/rel_9_predicates.json') as f:
                rel_prd_cats = json.load(f)
            rel_prd_cats.append('is')
            # rel APs
            rel_mAP = 0.
            for c in range(cls_num):
                rec, prec, ap = ap_eval(cls_image_ids[c], cls_dets[c], cls_gts[c], npos[c], True)
                rel_mAP += ap
                print('rel AP for class {}: {:.6f}'.format(rel_prd_cats[c], ap))
            rel_mAP /= float(cls_num)
            print('rel mAP: {:.6f}'.format(rel_mAP))
            # phr APs
            phr_mAP = 0.
            for c in range(cls_num):
                rec, prec, ap = ap_eval(cls_image_ids[c], cls_dets[c], cls_gts[c], npos[c], False)
                phr_mAP += ap
                print('phr AP for class {}: {:.6f}'.format(rel_prd_cats[c], ap))
            phr_mAP /= float(cls_num)
            print('phr mAP: {:.6f}'.format(phr_mAP))
            # total: 0.4 x rel_mAP + 0.2 x R@50 + 0.4 x phr_mAP
            final_score = 0.4 * rel_mAP + 0.2 * recalls[50] + 0.4 * phr_mAP
            print('final_score: {:.6f}'.format(final_score))
    
    topk_dets_f = os.path.join(output_dir, 'all_detections_topk.pkl')
    print('Saving all topk dets to ', topk_dets_f)
    with open(topk_dets_f, 'wb') as f:
        pickle.dump(topk_dets, f, pickle.HIGHEST_PROTOCOL)
    print('Done.')
    
    # save to csv format
    write_topk_dets_into_csv(topk_dets, output_dir)
    print('Done.')
    

def print_stats(recalls):
    # print('====================== ' + 'sgdet' + ' ============================')
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


def write_topk_dets_into_csv(topk_dets, output_dir):
    
    with open(cfg.DATA_DIR + '/openimages_v4/rel/rel_57_object_ids.json') as f:
        rel_obj_ids = json.load(f)
    with open(cfg.DATA_DIR + '/openimages_v4/rel/att_5_attribute_ids.json') as f:
        att_ids = json.load(f)
    with open(cfg.DATA_DIR + '/openimages_v4/rel/rel_9_predicates.json') as f:
        rel_prd_cats = json.load(f)
    rel_prd_cats.append('is')
    
    if cfg.TEST.DATASETS[0].find('kaggle') >= 0:
        with open(cfg.DATA_DIR + '/openimages_v4/rel/kaggle_test_images/test_image_sizes.json') as f:
            all_image_sizes = json.load(f)
    else:
        with open(cfg.DATA_DIR + '/openimages_v4/rel/all_image_sizes.json') as f:
            all_image_sizes = json.load(f)
        
    topk_dets_csv_f = os.path.join(output_dir, 'all_detections_topk.csv')
    print('Saving all topk dets csv to ', topk_dets_csv_f)
    
    if cfg.TEST.DATASETS[0].find('kaggle') >= 0:
        all_dets = {}
        
    with open(topk_dets_csv_f, 'w', newline='') as csvfile:
        csvwriter = csv.writer(csvfile, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
        if cfg.TEST.DATASETS[0].find('kaggle') >= 0:
            csvwriter.writerow(
                ['ImageID', 'PredictionString'])
        else:
            csvwriter.writerow(
                ['ImageID', 'Score', 'LabelName1', 'LabelName2',
                 'XMin1', 'XMax1', 'YMin1', 'YMax1',
                 'XMin2', 'XMax2', 'YMin2', 'YMax2',
                 'RelationshipLabel'])
        for det in tqdm(topk_dets):
            img_name = det['image'].split('/')[-1]
            img_id = img_name.split('.')[0]

            w = all_image_sizes[img_id][0]
            h = all_image_sizes[img_id][1]

            sbj_boxes = det['det_boxes_s_top']
            obj_boxes = det['det_boxes_o_top']
            sbj_labels = det['det_labels_s_top']
            obj_labels = det['det_labels_o_top']
            prd_labels = det['det_labels_p_top']
            det_scores = det['det_scores_top']

            if cfg.TEST.DATASETS[0].find('kaggle') >= 0:
                pred = ''
            for j in range(sbj_labels.shape[0]):
                sbj_box = sbj_boxes[j]
                obj_box = obj_boxes[j]
                sbj_lbl = sbj_labels[j]
                obj_lbl = obj_labels[j]
                prd_lbl = prd_labels[j]
                rel_score = det_scores[j]

                lbl_s = rel_obj_ids[sbj_lbl]
                lbl_p = rel_prd_cats[prd_lbl]
                if lbl_p == 'is':
                    lbl_o = att_ids[obj_lbl]
                else:
                    lbl_o = rel_obj_ids[obj_lbl]

                xmin1 = max(0., float(sbj_box[0]) / float(w))
                xmax1 = min(1., float(sbj_box[2]) / float(w))
                ymin1 = max(0., float(sbj_box[1]) / float(h))
                ymax1 = min(1., float(sbj_box[3]) / float(h))

                xmin2 = max(0., float(obj_box[0]) / float(w))
                xmax2 = min(1., float(obj_box[2]) / float(w))
                ymin2 = max(0., float(obj_box[1]) / float(h))
                ymax2 = min(1., float(obj_box[3]) / float(h))
                
                # remove singular boxes or boxes with one pixel width or height
                if xmin1 >= xmax1 or ymin1 >= ymax1 or xmin2 >= xmax2 or ymin2 >= ymax2:
                    continue

                if cfg.TEST.DATASETS[0].find('kaggle') >= 0:
                    pred += '{} {} {} {} {} {} {} {} {} {} {} {} '.format(
                        rel_score, lbl_s, xmin1, ymin1, xmax1, ymax1, lbl_o, xmin2, ymin2, xmax2, ymax2, lbl_p)
                else:
                    csvwriter.writerow([img_id, rel_score, lbl_s, lbl_o, xmin1, xmax1, ymin1, ymax1, xmin2, xmax2, ymin2, ymax2, lbl_p])
                    
            if cfg.TEST.DATASETS[0].find('kaggle') >= 0:
                pred_str = pred.strip()
                csvwriter.writerow([img_id, pred_str])
