"""
Main Training Script for Semi-Supervised Instance Segmentation on Historical Maps.
"""

import argparse
import os
import shutil
from ultralytics import YOLO
from src.dataset import process_single_image
from src.pseudo_label import generate_pseudo_labels

def parse_args():
    parser = argparse.ArgumentParser(description="Two-Stage Teacher-Student Self-Training Pipeline")
    parser.add_argument("--data", type=str, default="data.yaml", help="Path to dataset configuration YAML file")
    parser.add_argument("--strategy", type=str, choices=["none", "object", "patch"], default="none", help="Self-training pseudo-labeling strategy")
    parser.add_argument("--teacher-weights", type=str, default="results/exp_teacher/weights/best.pt", help="Path to trained teacher model checkpoint")
    parser.add_argument("--unlabeled-dir", type=str, default="demo_data/unlabeled_images", help="Path to raw full-sheet unlabeled map images")
    parser.add_argument("--epochs", type=int, default=50, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size for training")
    parser.add_argument("--imgsz", type=int, default=896, help="Tile resolution for model input")
    parser.add_argument("--conf-thresh", type=float, default=0.25, help="Confidence threshold for pseudo-label filtering")
    parser.add_argument("--project", type=str, default="results", help="Directory where training runs will be saved")
    parser.add_argument("--name", type=str, default="exp_run", help="Experiment run identifier")
    return parser.parse_args()

def main():
    args = parse_args()

    # Define standard YOLO dataset directories
    dataset_img_dir = "data/images"
    dataset_lbl_dir = "data/labels"
    processed_unlabeled_dir = "data/unlabeled_images"

    os.makedirs(dataset_img_dir, exist_ok=True)
    os.makedirs(dataset_lbl_dir, exist_ok=True)
    os.makedirs(processed_unlabeled_dir, exist_ok=True)

    # Stage 1: Standard Supervised Training (Teacher Model)
    if args.strategy == "none":
        print("🚀 Starting Stage 1: Supervised Teacher Model Training...")
        model = YOLO("yolo11s-seg.pt")
        model.train(
            data=args.data,
            epochs=args.epochs,
            batch=args.batch_size,
            imgsz=args.imgsz,
            project=args.project,
            name=args.name,
            exist_ok=True
        )
        print(f"✅ Teacher training complete. Weights saved under {args.project}/{args.name}/weights/best.pt")

    # Stage 2: Semi-Supervised Self-Training (Student Model)
    else:
        print(f"🔄 Starting Stage 2: Self-Training with Strategy: {args.strategy.upper()}...")

        if not os.path.exists(args.teacher_weights):
            raise FileNotFoundError(f"❌ Teacher checkpoint not found at: {args.teacher_weights}")

        if not os.path.exists(args.unlabeled_dir):
            raise FileNotFoundError(f"❌ Raw unlabeled directory not found at: {args.unlabeled_dir}")

        # Step 1: Tile raw maps from demo_data into data/unlabeled_images using dataset.py
        print(f"🧩 Step 1: Tiling raw maps from '{args.unlabeled_dir}' into '{processed_unlabeled_dir}'...")
        valid_exts = ('.jpg', '.jpeg', '.png')
        raw_image_names = [
            os.path.splitext(f)[0]
            for f in os.listdir(args.unlabeled_dir)
            if f.lower().endswith(valid_exts)
        ]

        dummy_label_dir = "data/temp_unlabeled_labels"
        os.makedirs(dummy_label_dir, exist_ok=True)

        for img_name in raw_image_names:
            process_single_image(
                image_name=img_name,
                images_dir=args.unlabeled_dir,
                labels_dir=args.unlabeled_dir,
                output_images_dir=processed_unlabeled_dir,
                output_labels_dir=dummy_label_dir,
                tile_size=args.imgsz
            )

        # Cleanup temporary labels folder created during tiling
        shutil.rmtree(dummy_label_dir, ignore_errors=True)

        tiled_images = [f for f in os.listdir(processed_unlabeled_dir) if f.lower().endswith(valid_exts)]

        if not tiled_images:
            print(f"⚠️ No valid image tiles found in '{processed_unlabeled_dir}'. Proceeding without pseudo-labels.")
        else:
            # Step 2: Generate pseudo-labels using selected strategy into a unified folder
            pseudo_dir = "data/pseudo_labels"
            
            # Clean up existing pseudo_labels directory if present to avoid mixing runs
            if os.path.exists(pseudo_dir):
                shutil.rmtree(pseudo_dir)

            print(f"⚙️ Step 2: Generating pseudo-labels for {len(tiled_images)} tiles using strategy '{args.strategy.upper()}'...")
            teacher_model = YOLO(args.teacher_weights)

            generate_pseudo_labels(
                images_list=tiled_images,
                model=teacher_model,
                threshold=args.conf_thresh,
                unlabeled_dir=processed_unlabeled_dir,
                pseudo_dir=pseudo_dir,
                imgsz=args.imgsz,
                strategy=args.strategy
            )

            # Step 3: Merge pseudo-labels (.txt) and copy corresponding image tiles into dataset folders
            if os.path.exists(pseudo_dir):
                pseudo_files = [f for f in os.listdir(pseudo_dir) if f.endswith('.txt')]

                merged_count = 0
                for txt_file in pseudo_files:
                    src_txt = os.path.join(pseudo_dir, txt_file)
                    dst_txt = os.path.join(dataset_lbl_dir, txt_file)

                    # Copy pseudo-label file to data/labels
                    shutil.copy(src_txt, dst_txt)

                    # Copy matching tile image to data/images
                    base_name = os.path.splitext(txt_file)[0]
                    for ext in valid_exts:
                        img_name = base_name + ext
                        src_img = os.path.join(processed_unlabeled_dir, img_name)
                        if os.path.exists(src_img):
                            shutil.copy(src_img, os.path.join(dataset_img_dir, img_name))
                            break

                    merged_count += 1

                print(f"✅ Step 3: Successfully merged {merged_count} {args.strategy.upper()} pseudo-labels into '{dataset_lbl_dir}' and synced tiles to '{dataset_img_dir}'.")
            else:
                print(f"ℹ️ Step 3: No pseudo-labels met the confidence threshold ({args.conf_thresh}) under strategy '{args.strategy.upper()}'.")

        # Step 4: Train Student model on integrated dataset
        print("🎓 Step 4: Training Student model on augmented dataset...")
        student_model = YOLO("yolo11s-seg.pt")
        student_model.train(
            data=args.data,
            epochs=args.epochs,
            batch=args.batch_size,
            imgsz=args.imgsz,
            project=args.project,
            name=args.name,
            exist_ok=True
        )
        print(f"🎉 Self-training complete! Student model saved at {args.project}/{args.name}/weights/best.pt")


if __name__ == "__main__":
    main()