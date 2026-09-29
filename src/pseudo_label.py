import os
import json
import shutil
import cv2
import numpy as np
from PIL import Image
from shapely.geometry import Polygon, MultiPolygon

def mask_to_yolo_polygons(mask, width, height, cls_id, min_area=40, epsilon_ratio=0.01):
    """Convert binary mask into normalized YOLO polygon annotations."""
    labels = []

    # Ensure mask values are 0-255 uint8
    if mask.max() == 1:
        mask = (mask * 255).astype(np.uint8)

    # Extract external contours
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
            if not poly.is_valid or len(poly.exterior.coords) < 4:
                continue
        except ValueError:
            continue

        area = cv2.contourArea(cnt)
        if area < min_area:
            continue

        # Enforce clockwise orientation
        if cv2.contourArea(cnt, oriented=True) < 0:
            cnt = cnt[::-1]

        # PolyDP approximation
        epsilon = epsilon_ratio * cv2.arcLength(cnt, True)
        cnt = cv2.approxPolyDP(cnt, epsilon, True)
        pts = cnt.reshape(-1, 2)

        # Shapely geometry repair
        poly = Polygon(pts)
        if not poly.is_valid:
            poly = poly.buffer(0)

        if poly.is_empty or not poly.is_valid:
            continue

        # Handle MultiPolygon outputs
        if isinstance(poly, MultiPolygon):
            polys = [p for p in poly.geoms if len(p.exterior.coords) >= 4 and p.area >= min_area]
        else:
            polys = [poly] if len(poly.exterior.coords) >= 4 and poly.area >= min_area else []

        # Convert to normalized YOLO coordinates (0-1)
        for p in polys:
            xs, ys = p.exterior.xy
            coords = []
            for x, y in zip(xs, ys):
                coords.append(x / width)
                coords.append(y / height)

            if len(coords) >= 6:
                line = f"{cls_id} " + " ".join(f"{v:.6f}" for v in coords)
                labels.append(line + "\n")

    return labels
    
def validate_and_clean_labels(label_path):
    """
    Validates polygon coordinates in a YOLO label file and removes invalid lines.
    """
    if not os.path.exists(label_path):
        return

    valid_lines = []
    with open(label_path, 'r') as f:
        lines = f.readlines()

    for line in lines:
        parts = line.strip().split()
        if len(parts) < 7:  # class_id + at least 3 points (6 coords)
            continue

        try:
            coords = list(map(float, parts[1:]))
        except ValueError:
            continue

        # Check coordinate pairs are within valid normalized range [0.0, 1.0]
        if all(0.0 <= c <= 1.0 for c in coords):
            valid_lines.append(line)

    if valid_lines:
        with open(label_path, 'w') as f:
            f.writelines(valid_lines)
    else:     
        os.remove(label_path) # Remove empty label file if no valid polygons exist

def prepare_pseudo_dirs(pseudo_O_dir="data/pseudo_O", pseudo_P_dir="data/pseudo_P"):
    """
    Prepare clean temporary directories for pseudo-labels.
    """
    if os.path.exists(pseudo_O_dir):
        shutil.rmtree(pseudo_O_dir)
    os.makedirs(pseudo_O_dir, exist_ok=True)

    if os.path.exists(pseudo_P_dir):
        shutil.rmtree(pseudo_P_dir)
    os.makedirs(pseudo_P_dir, exist_ok=True)

def generate_pseudo_labels(images_list, model, threshold, unlabeled_dir,
                           pseudo_dir="data/pseudo_labels",
                           imgsz=896, strategy="object"):
    """
    Generate Strategy-O (Object-level) and Strategy-P (Patch-level) pseudo-labels 
    for unlabeled map sheets using a trained teacher model.
    """
    valid_exts = ('.jpg', '.jpeg', '.png',)

    for img_name in images_list:
        if not img_name.lower().endswith(valid_exts):
            continue

        img_path = os.path.join(unlabeled_dir, img_name)

        # For Strategy-P, use low conf in predict to catch noisy/uncertain predictions in the tile
        predict_conf = 0.1 if strategy == "patch" else threshold
        results = model.predict(source=img_path, conf=predict_conf, imgsz=imgsz, verbose=False)

        if not results or len(results[0]) == 0:
            continue

        result = results[0]
        if result.masks is None:
            continue

        confs = result.boxes.conf.cpu().numpy()
        classes = result.boxes.cls.cpu().numpy().astype(int)
        masks = result.masks.xyn  # Normalized polygon coordinates [(N, 2)]

        kept_polygons = []

        # ----- Strategy 1: Object-level Filtering (Strategy-O) -----
        if strategy == "object":
            for cls_id, conf, mask in zip(classes, confs, masks):
                if conf >= threshold and len(mask) >= 3:
                    coords_str = " ".join([f"{x:.6f} {y:.6f}" for x, y in mask])
                    kept_polygons.append(f"{cls_id} {coords_str}")

            if len(kept_polygons) > 0:
                os.makedirs(pseudo_dir, exist_ok=True)
                txt_name = os.path.splitext(img_name)[0] + ".txt"
                pseudo_path = os.path.join(pseudo_dir, txt_name)

                content = "\n".join(kept_polygons).strip() + "\n"
                with open(pseudo_path, 'w') as f:
                    f.write(content)

                validate_and_clean_labels(pseudo_path)

        # ----- Strategy 2: Patch-level Filtering (Strategy-P) -----
        elif strategy == "patch":
            # Check if ANY predicted object in this tile falls below the strict threshold
            has_low_conf_object = any(conf < threshold for conf in confs)

            # Keep tile ONLY if there is at least one prediction AND no low-confidence clutter
            if len(confs) > 0 and not has_low_conf_object:
                for cls_id, conf, mask in zip(classes, confs, masks):
                    if len(mask) >= 3:
                        coords_str = " ".join([f"{x:.6f} {y:.6f}" for x, y in mask])
                        kept_polygons.append(f"{cls_id} {coords_str}")

                if len(kept_polygons) > 0:
                    os.makedirs(pseudo_dir, exist_ok=True)
                    txt_name = os.path.splitext(img_name)[0] + ".txt"
                    pseudo_path = os.path.join(pseudo_dir, txt_name)

                    content = "\n".join(kept_polygons).strip() + "\n"
                    with open(pseudo_path, 'w') as f:
                        f.write(content)

                    validate_and_clean_labels(pseudo_path)