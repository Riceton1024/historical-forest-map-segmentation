"""
Pseudo-Labeling Generator for Semi-Supervised Instance Segmentation
Implements Strategy-O (Object-level) and Strategy-P (Patch-level) filtering mechanisms.
"""

import os
import json
import cv2
import numpy as np
from PIL import Image
from shapely.geometry import Polygon, MultiPolygon

def mask_to_yolo_polygons(mask, width, height, cls_id=0, min_area=40, epsilon_ratio=0.01):
    """Converts a binary mask into YOLO-formatted normalized polygon coordinates."""
    labels = []
    if mask.max() == 1:
        mask = (mask * 255).astype(np.uint8)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for cnt in contours:
        if len(cnt) < 3 or cv2.contourArea(cnt) < min_area:
            continue
        
        epsilon = epsilon_ratio * cv2.arcLength(cnt, True)
        cnt = cv2.approxPolyDP(cnt, epsilon, True)
        if len(cnt) < 3:
            continue

        try:
            poly = Polygon(cnt.reshape(-1, 2))
            if not poly.is_valid:
                poly = poly.buffer(0)
        except Exception:
            continue

        if poly.is_empty or not poly.is_valid:
            continue

        polys = [p for p in (poly.geoms if isinstance(poly, MultiPolygon) else [poly]) 
                 if len(p.exterior.coords) >= 4 and p.area >= min_area]

        for p in polys:
            xs, ys = p.exterior.xy
            coords = []
            for x, y in zip(xs, ys):
                coords.append(x / width)
                coords.append(y / height)

            if len(coords) >= 6:
                labels.append(f"{cls_id} " + " ".join(f"{v:.6f}" for v in coords) + "\n")

    return labels

def generate_pseudo_labels(images_list, model, img_dir, pseudo_A_dir, pseudo_B_dir, 
                           conf_threshold=0.25, imgsz=896):
    """
    Generates Strategy-O (Object-level) and Strategy-P (Patch-level) pseudo-labels.
    
    Args:
        images_list (list): List of image filenames.
        model (YOLO): Trained YOLO model.
        conf_threshold (float): Confidence threshold for pseudo-label filtering.
    """
    os.makedirs(pseudo_A_dir, exist_ok=True)
    os.makedirs(pseudo_B_dir, exist_ok=True)

    for img_name in images_list:
        img_path = os.path.join(img_dir, img_name)
        if not os.path.exists(img_path):
            continue

        with Image.open(img_path) as img:
            width, height = img.size

        results = model.predict(source=img_path, imgsz=imgsz, retina_masks=True, verbose=False)
        masks = results[0].masks
        boxes = results[0].boxes
        
        if masks is None or boxes is None:
            continue

        confs = boxes.conf.cpu().numpy()
        cls_ids = boxes.cls.cpu().numpy()
        kept_A = []

        # ----- Strategy-O (Object-level) -----
        for i in range(len(masks)):
            if confs[i] < conf_threshold:
                continue
            
            mask_data = masks.data[i].cpu().numpy().astype(np.uint8)
            mask_np = cv2.morphologyEx(mask_data, cv2.MORPH_CLOSE, np.ones((2, 2), np.uint8))
            
            polys = mask_to_yolo_polygons(mask_np, width, height, int(cls_ids[i]))
            kept_A.extend(polys)

        txt_name = os.path.splitext(img_name)[0] + ".txt"

        if len(kept_A) > 0:
            with open(os.path.join(pseudo_A_dir, txt_name), 'w') as f:
                f.writelines(kept_A)

        # ----- Strategy-P (Patch-level) -----
        is_patch_valid = len(confs) > 0 and min(confs) >= conf_threshold
        if is_patch_valid and len(kept_A) > 0:
            with open(os.path.join(pseudo_B_dir, txt_name), 'w') as f:
                f.writelines(kept_A)