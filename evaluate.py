"""
Main Evaluation Script
Loads trained models, performs whole-map reconstruction, and saves metric reports.
"""

import argparse
import os
import re
import cv2
import numpy as np
from ultralytics import YOLO
from src.metrics import calculate_iou, evaluate_instance_segmentation

def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate Instance Segmentation on Historical Maps")
    parser.add_argument("--model", type=str, required=True, help="Path to trained .pt model weights")
    parser.add_argument("--images-dir", type=str, required=True, help="Path to directory containing full test map images")
    parser.add_argument("--labels-dir", type=str, required=True, help="Path to directory containing full test map labels")
    parser.add_argument("--tiles-images-dir", type=str, default="data/images", help="Path to directory containing image tiles")
    parser.add_argument("--test-txt", type=str, default=None, help="[Optional] Path to txt file listing specific map names")
    parser.add_argument("--output-txt", type=str, default="results/eval_report.txt", help="Path to save evaluation report")
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold")
    parser.add_argument("--iou", type=float, default=0.7, help="NMS IoU threshold")
    return parser.parse_args()

def process_true_mask(true_mask_path, ori_height, ori_width):
    """
    Parses YOLO format txt label file and draws ground truth binary mask.
    """
    true_mask = np.zeros((ori_height, ori_width), dtype=np.uint8)
    if not os.path.exists(true_mask_path):
        return true_mask

    with open(true_mask_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    for line in lines:
        parts = line.strip().split()
        if len(parts) < 5:
            continue
        coords = np.array([float(x) for x in parts[1:]]).reshape(-1, 2)
        coords[:, 0] *= ori_width
        coords[:, 1] *= ori_height
        pts = coords.astype(np.int32)
        cv2.fillPoly(true_mask, [pts], 1)

    return true_mask

def get_combined_mask(base_name, slice_image_path, model, conf, iou, slice_size=896, img_size=896, ori_height=0, ori_width=0):
    combined_mask_total = np.zeros((ori_height, ori_width), dtype=np.uint8)
    current_individual_masks = []

    for img in slice_image_path:
        img_slice = cv2.imread(img)
        if img_slice is None:
            continue

        results = model.predict(img_slice, conf=conf, iou=iou, verbose=False, retina_masks=True)
        masks = results[0].masks
        mask_class = results[0].boxes.cls if results[0].boxes is not None else []
        boxes = results[0].boxes

        combined_mask = np.zeros((img_size, img_size), dtype=np.uint8)

        if masks:
            for idx, (mask, cls_id) in enumerate(zip(masks, mask_class)):
                single_mask = mask.data[0].cpu().numpy()
                combined_mask = np.maximum(combined_mask, single_mask)

                current_individual_masks.append({
                    'mask': single_mask,
                    'class_id': cls_id.item(),
                    'bbox': boxes.xyxy[idx].cpu().numpy().tolist(),
                    'image_name': img,
                    'mask_name': img.replace('/images/', '/labels/').replace('.jpg', '.txt'),
                    'confidence': results[0].boxes.conf[idx].item(),
                })

        combined_mask_resized = cv2.resize(combined_mask, (slice_size, slice_size), interpolation=cv2.INTER_NEAREST)
        match = re.search(rf"{base_name}_(\d+)_(\d+)\.jpg", img)
        if match:
            row_index, col_index = int(match.group(1)), int(match.group(2))
            start_y, start_x = col_index * slice_size, row_index * slice_size
            h, w = min(slice_size, ori_height - start_y), min(slice_size, ori_width - start_x)

            if h > 0 and w > 0:
                mask_cliped = combined_mask_resized[:h, :w]
                combined_mask_total[start_y:start_y + h, start_x:start_x + w] = mask_cliped

    return combined_mask_total, current_individual_masks

def predict_and_reconstruct_single_image(base_name, model, images_dir, labels_dir, tiles_images_dir, conf=0.25, iou=0.7):
    ori_image_path = os.path.join(images_dir, f"{base_name}.jpg")
    ori_image = cv2.imread(ori_image_path)
    if ori_image is None:
        print(f"❌ Warning: Could not read image at {ori_image_path}")
        return None

    ori_height, ori_width = ori_image.shape[:2]

    true_mask_path = os.path.join(labels_dir, f"{base_name}.txt")
    true_mask = process_true_mask(true_mask_path, ori_height, ori_width)

    slice_image_path = [
        os.path.join(tiles_images_dir, f) 
        for f in os.listdir(tiles_images_dir) 
        if f.startswith(f"{base_name}_") and f.endswith(".jpg")
    ]

    if not slice_image_path:
        print(f"❌ Warning: No tile images found for {base_name} in {tiles_images_dir}")
        return None

    combined_mask_total, current_individual_masks = get_combined_mask(
        base_name, slice_image_path, model, conf, iou, 
        slice_size=896, img_size=896, ori_height=ori_height, ori_width=ori_width
    )

    iou_score = calculate_iou(combined_mask_total, true_mask)

    return {
        'original_image': ori_image,
        'true_mask': true_mask,
        'individual_masks': current_individual_masks,
        'IoU': iou_score,
    }

def main():
    args = parse_args()
    if not os.path.exists(args.model):
        print(f"❌ Error: Model checkpoint not found at {args.model}")
        return

    # Determine target map names
    if args.test_txt and os.path.exists(args.test_txt):
        print(f"📖 Reading test map names from: {args.test_txt}")
        with open(args.test_txt, 'r', encoding='utf-8') as f:
            test_map_names = [line.strip() for line in f if line.strip()]
    else:
        print(f"📂 Scanning images directory: {args.images_dir}")
        valid_exts = ('.jpg', '.jpeg', '.png', '.tif', '.tiff')
        test_map_names = [
            os.path.splitext(f)[0] 
            for f in os.listdir(args.images_dir) 
            if f.lower().endswith(valid_exts)
        ]

    if not test_map_names:
        print("❌ Error: No test map images found.")
        return

    print(f"🚀 Target evaluation maps ({len(test_map_names)}): {test_map_names}")

    model = YOLO(args.model)
    results_list = []

    for test_name in test_map_names:
        initial_result = predict_and_reconstruct_single_image(
            test_name, model, args.images_dir, args.labels_dir, args.tiles_images_dir,
            conf=args.conf, iou=args.iou
        )
        if initial_result is None:
            continue

        metrics = evaluate_instance_segmentation(initial_result)
        metrics['image_name'] = test_name
        results_list.append(metrics)

    os.makedirs(os.path.dirname(args.output_txt), exist_ok=True)
    with open(args.output_txt, 'w', encoding='utf-8') as f:
        f.write("image_name\tTotal TP\tTotal FP\tTotal FN\tPrecision\tRecall\tF1-Score\tArea Weighted Mean IoU (TP)\tmAP@50\tmAP\toverall_iou\n")
        for res in results_list:
            line = (
                f"{res['image_name']}\t"
                f"{res['tp']}\t"
                f"{res['fp']}\t"
                f"{res['fn']}\t"
                f"{res['precision']:.2f}\t"
                f"{res['recall']:.2f}\t"
                f"{res['f1_score']:.2f}\t"
                f"{res['area_weighted_mean_iou']*100:.1f}%\t"
                f"{res['ap_50']*100:.1f}%\t"
                f"{res['mAP']*100:.1f}%\t"
                f"{res['overall_iou']*100:.1f}%\n"
            )
            f.write(line)

    print(f"✅ Evaluation report saved to {args.output_txt}")

if __name__ == "__main__":
    main()