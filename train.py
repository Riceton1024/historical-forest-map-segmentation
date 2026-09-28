"""
Main Training Script for Semi-Supervised Instance Segmentation on Historical Maps.
"""

import argparse
import os
from ultralytics import YOLO
from src.pseudo_label import generate_pseudo_labels

def parse_args():
    parser = argparse.ArgumentParser(description="Train YOLOv11-seg with Pseudo-Labeling")
    parser.add_argument("--data", type=str, default="data.yaml", help="Path to data.yaml")
    parser.add_argument("--strategy", type=str, choices=["none", "object", "patch"], default="none",
                        help="Pseudo-label strategy: 'object' (Strategy-O) or 'patch' (Strategy-P)")
    parser.add_argument("--epochs", type=int, default=200, help="Number of epochs")
    parser.add_argument("--batch-size", type=int, default=2, help="Batch size")
    parser.add_argument("--imgsz", type=int, default=896, help="Image size")
    parser.add_argument("--conf-thresh", type=float, default=0.25, help="Confidence threshold")
    parser.add_argument("--project", type=str, default="results", help="Project path")
    parser.add_argument("--name", type=str, default="exp_ssl", help="Experiment name")
    return parser.parse_args()

def main():
    args = parse_args()
    
    # 1. Load Pretrained YOLO Model
    model = YOLO("yolo11s-seg.pt")

    # 2. Start Training
    print(f"Starting training with Strategy: {args.strategy}")
    model.train(
        data=args.data,
        epochs=args.epochs,
        batch=args.batch_size,
        imgsz=args.imgsz,
        project=args.project,
        name=args.name,
        mask_ratio=4,
        dropout=0.1,
        val=True,
        plots=True
    )

if __name__ == "__main__":
    main()