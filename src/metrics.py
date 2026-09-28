"""
Evaluation Metrics & Instance Matching
Includes IoU computation, Hungarian Matching, AW-IoU, and mAP calculation.
"""

import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from collections import defaultdict
from scipy.optimize import linear_sum_assignment

def calculate_iou(pred_mask, true_mask):
    """Calculates Intersection over Union (IoU) between two binary masks."""
    pred_mask = (pred_mask > 0).astype(np.uint8)
    true_mask = (true_mask > 0).astype(np.uint8)
    intersection = np.logical_and(pred_mask, true_mask).sum()
    union = np.logical_or(pred_mask, true_mask).sum()
    return 1.0 if union == 0 and intersection == 0 else (intersection / union if union > 0 else 0.0)

def coco_txt_to_mask(txt_path, img_shape):
    """Converts YOLO polygon txt into binary masks."""
    masks = []
    img_height, img_width = img_shape
    if not os.path.exists(txt_path):
        return masks

    with open(txt_path, 'r', encoding='utf-8') as f:
        for line in f:
            parts = list(map(float, line.strip().split()))
            if len(parts) < 5:
                continue
            coords = np.array(parts[1:]).reshape(-1, 2)
            coords[:, 0] = np.clip(coords[:, 0] * img_width, 0, img_width - 1)
            coords[:, 1] = np.clip(coords[:, 1] * img_height, 0, img_height - 1)
            
            mask = np.zeros(img_shape, dtype=np.uint8)
            cv2.fillPoly(mask, [coords.astype(np.int32)], 1)
            masks.append(mask.astype(bool))
    return masks

def hungarian_match(pred_masks, gt_masks):
    """Performs Hungarian matching based on IoU matrix."""
    if len(pred_masks) == 0 or len(gt_masks) == 0:
        return [], np.zeros((len(pred_masks), len(gt_masks)))

    iou_matrix = np.zeros((len(pred_masks), len(gt_masks)))
    for i, pred in enumerate(pred_masks):
        for j, gt in enumerate(gt_masks):
            iou_matrix[i, j] = calculate_iou(pred, gt)

    row_ind, col_ind = linear_sum_assignment(-iou_matrix)
    matched_pairs = []
    for r, c in zip(row_ind, col_ind):
        matched_pairs.append((r, c, iou_matrix[r, c]))
    return matched_pairs, iou_matrix

def evaluate_instance_segmentation(initial_result, iou_thresholds=np.arange(0.5, 1.0, 0.05), visualize=False):
    """Evaluates Precision, Recall, F1, AW-IoU, AP@50, and mAP across IoU thresholds."""
    ap_results = defaultdict(list)
    total_tp, total_fp, total_fn = 0, 0, 0
    weighted_iou_sum = 0.0
    total_gt_area = 0.0

    grouped_by_mask_name = defaultdict(list)
    for item in initial_result['individual_masks']:
        grouped_by_mask_name[item['mask_name']].append(item)

    for mask_txt_path, preds in grouped_by_mask_name.items():
        pred_groups = [{
            'mask': item['mask'].astype(bool),
            'confidence': item['confidence'],
            'class_id': item.get('class_id', 0)
        } for item in preds]

        if len(pred_groups) == 0:
            continue

        gt_masks = coco_txt_to_mask(mask_txt_path, pred_groups[0]['mask'].shape)
        pred_masks = [item['mask'] for item in pred_groups]

        if len(gt_masks) == 0:
            total_fp += len(pred_masks)
            continue
        elif len(pred_masks) == 0:
            total_fn += len(gt_masks)
            continue

        matched_pairs, _ = hungarian_match(pred_masks, gt_masks)

        matched_pred_indices = set()
        matched_gt_indices = set()
        tp_per_image = 0

        # Primary evaluation at IoU >= 0.5
        for pred_idx, gt_idx, iou in matched_pairs:
            if iou >= 0.5:
                tp_per_image += 1
                matched_pred_indices.add(pred_idx)
                matched_gt_indices.add(gt_idx)

                # Accumulate AW-IoU
                gt_area = gt_masks[gt_idx].sum()
                weighted_iou_sum += iou * gt_area
                total_gt_area += gt_area

        fp_per_image = len(pred_masks) - len(matched_pred_indices)
        fn_per_image = len(gt_masks) - len(matched_gt_indices)

        total_tp += tp_per_image
        total_fp += fp_per_image
        total_fn += fn_per_image

        # Multi-threshold precision evaluation for mAP
        for iou_thresh in iou_thresholds:
            matched_pred = set()
            tp = 0
            for pred_idx, gt_idx, iou in matched_pairs:
                if iou >= iou_thresh:
                    tp += 1
                    matched_pred.add(pred_idx)

            fp = len(pred_masks) - len(matched_pred)
            precision_thresh = tp / (tp + fp + 1e-8)
            ap_results[iou_thresh].append(precision_thresh)

    # === Overall Summary Metrics ===
    precision = total_tp / (total_tp + total_fp + 1e-8)
    recall = total_tp / (total_tp + total_fn + 1e-8)
    f1_score = 2 * precision * recall / (precision + recall + 1e-8)
    area_weighted_mean_iou = weighted_iou_sum / total_gt_area if total_gt_area > 0 else 0.0

    ap_05 = np.mean(ap_results[0.5]) if 0.5 in ap_results and ap_results[0.5] else 0.0

    ap_values = []
    for thresh in iou_thresholds:
        ap = np.mean(ap_results[thresh]) if ap_results[thresh] else 0.0
        ap_values.append(ap)
    mAP = np.mean(ap_values) if ap_values else ap_05

    return {
        "tp": total_tp,
        "fp": total_fp,
        "fn": total_fn,
        "precision": precision,
        "recall": recall,
        "f1_score": f1_score,
        "area_weighted_mean_iou": area_weighted_mean_iou,
        "ap_50": ap_05,
        "mAP": mAP,
        "overall_iou": initial_result['IoU'],
    }