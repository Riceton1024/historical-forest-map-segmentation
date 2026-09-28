"""
Dataset Preparation & Tile Processing
Handles splitting high-resolution historical maps into manageable image patches.
"""

import os
import argparse
import shutil
import cv2
import numpy as np
from shapely.geometry import Polygon, box

def adjust_polygon_for_tile(label_file, x_offset, y_offset, tile_size, image_width, image_height):
    """Adjusts polygon coordinates for a cropped tile."""
    new_labels = []
    tile_left = x_offset * tile_size
    tile_top = y_offset * tile_size
    tile_box = box(tile_left, tile_top, tile_left + tile_size, tile_top + tile_size)

    if not os.path.exists(label_file):
        return new_labels

    with open(label_file, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if not parts:
                continue
            class_id = int(parts[0])
            if class_id not in (0, 1):
                continue

            coords = list(map(float, parts[1:]))
            pts = [(coords[i] * image_width, coords[i+1] * image_height) for i in range(0, len(coords), 2)]
            poly = Polygon(pts)

            if not poly.is_valid:
                poly = poly.buffer(0)

            clipped = poly.intersection(tile_box)
            if clipped.is_empty or not clipped.is_valid:
                continue

            polys = [clipped] if isinstance(clipped, Polygon) else list(clipped.geoms) if clipped.geom_type == "MultiPolygon" else []

            for cpoly in polys:
                if cpoly.is_empty or not cpoly.is_valid:
                    continue
                xs, ys = cpoly.exterior.xy
                clipped_coords = []
                for x, y in zip(xs, ys):
                    rel_x = max(0.0, min(1.0, (x - tile_left) / tile_size))
                    rel_y = max(0.0, min(1.0, (y - tile_top) / tile_size))
                    clipped_coords.extend([rel_x, rel_y])

                if len(clipped_coords) >= 6:
                    new_labels.append(f"{class_id} " + " ".join(f"{v:.6f}" for v in clipped_coords) + "\n")

    return new_labels

def process_single_image(image_name, images_dir, labels_dir, output_images_dir, output_labels_dir, tile_size=896):
    """Crops a single large map into tiles and saves updated YOLO annotations."""
    image_path = os.path.join(images_dir, f"{image_name}.jpg")
    label_path = os.path.join(labels_dir, f"{image_name}.txt")

    image = cv2.imread(image_path)
    if image is None:
        return

    h, w, _ = image.shape
    num_tiles_x = (w + tile_size - 1) // tile_size
    num_tiles_y = (h + tile_size - 1) // tile_size

    for y_offset in range(num_tiles_y):
        for x_offset in range(num_tiles_x):
            start_x, start_y = x_offset * tile_size, y_offset * tile_size
            end_x, end_y = min(start_x + tile_size, w), min(start_y + tile_size, h)

            tile_image = image[start_y:end_y, start_x:end_x]
            padded_image = np.zeros((tile_size, tile_size, 3), dtype=np.uint8)
            padded_image[:end_y - start_y, :end_x - start_x] = tile_image

            tile_filename = f"{image_name}_{x_offset}_{y_offset}"
            cv2.imwrite(os.path.join(output_images_dir, f"{tile_filename}.jpg"), padded_image)

            new_labels = adjust_polygon_for_tile(label_path, x_offset, y_offset, tile_size, w, h)
            with open(os.path.join(output_labels_dir, f"{tile_filename}.txt"), 'w') as f:
                f.writelines(new_labels)

def process_all_maps(images_dir, labels_dir, output_dir, tile_size=896):
    """Iterate over all images in the input directory and apply tile processing."""
    output_images_dir = os.path.join(output_dir, "images")
    output_labels_dir = os.path.join(output_dir, "labels")
    
    os.makedirs(output_images_dir, exist_ok=True)
    os.makedirs(output_labels_dir, exist_ok=True)

    if not os.path.exists(images_dir):
        print(f"Error: Input directory not found: {images_dir}")
        return

    image_files = [
        os.path.splitext(f)[0] 
        for f in os.listdir(images_dir) 
        if f.lower().endswith(('.jpg', '.jpeg', '.png'))
    ]
    
    print(f"Processing {len(image_files)} historical map images...")
    for img_name in image_files:
        process_single_image(
            image_name=img_name,
            images_dir=images_dir,
            labels_dir=labels_dir,
            output_images_dir=output_images_dir,
            output_labels_dir=output_labels_dir,
            tile_size=tile_size
        )
        print(f"  Processed: {img_name}")
    
    print("All images tiled successfully.")

def main():
    parser = argparse.ArgumentParser(description="Tile historical map images and labels into patches.")
    parser.add_argument("--images-dir", type=str, required=True, help="Path to input full map images")
    parser.add_argument("--labels-dir", type=str, required=True, help="Path to input full map labels")
    parser.add_argument("--output-dir", type=str, default="data", help="Path to output data folder")
    parser.add_argument("--tile-size", type=int, default=896, help="Tile size in pixels")
    args = parser.parse_args()

    process_all_maps(
        images_dir=args.images_dir,
        labels_dir=args.labels_dir,
        output_dir=args.output_dir,
        tile_size=args.tile_size
    )

if __name__ == "__main__":
    main()