#!/usr/bin/env python3
"""
prepare_v2_annotations.py
─────────────────────────
Prepares annotation structures and pre-labeling manifests for RipVIS V2.

Sets up:
- COCO-format empty/template annotations: data/V2/annotations/coco_v2.json
- YOLOv8 segmentation directory structure: data/V2/annotations/yolo/
- data.yaml specification for YOLOv8 training/validation
- Automatic image registration for Label Studio / CVAT / Roboflow import
"""

import argparse
import datetime
import json
import logging
from pathlib import Path

from PIL import Image

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger("RipVIS_V2_Annotator")


def setup_annotation_structure(base_dir: Path):
    clean_img_dir = base_dir / "cleaned" / "images"
    ann_dir = base_dir / "annotations"
    yolo_dir = ann_dir / "yolo"
    yolo_img_dir = yolo_dir / "images"
    yolo_lbl_dir = yolo_dir / "labels"

    ann_dir.mkdir(parents=True, exist_ok=True)
    yolo_dir.mkdir(parents=True, exist_ok=True)
    yolo_img_dir.mkdir(parents=True, exist_ok=True)
    yolo_lbl_dir.mkdir(parents=True, exist_ok=True)

    images = sorted(list(clean_img_dir.glob("*.jpg")) + list(clean_img_dir.glob("*.png")))
    logger.info(f"Registering {len(images)} cleaned images into annotation structure...")

    # Build COCO template
    coco_data = {
        "info": {
            "description": "RipVIS V2 Dataset — Rip Currents & Ocean Swimmers Instance Segmentation",
            "url": "https://ripvis.ai",
            "version": "2.0",
            "year": datetime.date.today().year,
            "date_created": datetime.date.today().isoformat()
        },
        "licenses": [{"id": 1, "name": "Research and Academic Use"}],
        "categories": [
            {"id": 0, "name": "rip_current", "supercategory": "hazard"},
            {"id": 1, "name": "swimmer", "supercategory": "person"}
        ],
        "images": [],
        "annotations": []
    }

    # Label Studio Task Import JSON
    label_studio_tasks = []

    for img_id, img_p in enumerate(images):
        try:
            with Image.open(img_p) as im:
                w, h = im.size
        except Exception:
            continue

        coco_data["images"].append({
            "id": img_id,
            "file_name": img_p.name,
            "width": w,
            "height": h,
            "date_captured": datetime.date.today().isoformat()
        })

        label_studio_tasks.append({
            "data": {
                "image": f"/data/local-files/?d=V2/cleaned/images/{img_p.name}",
                "original_filename": img_p.name,
                "width": w,
                "height": h
            }
        })

    # Save COCO Template
    coco_path = ann_dir / "coco_v2_template.json"
    with open(coco_path, "w") as f:
        json.dump(coco_data, f, indent=2)
    logger.info(f"Saved COCO annotation template to: {coco_path}")

    # Save Label Studio Tasks
    ls_path = ann_dir / "label_studio_tasks.json"
    with open(ls_path, "w") as f:
        json.dump(label_studio_tasks, f, indent=2)
    logger.info(f"Saved Label Studio task import list to: {ls_path}")

    # Create data.yaml for YOLOv8
    yaml_content = f"""# RipVIS V2 Dataset Config
path: {yolo_dir.resolve()}
train: images/train
val: images/val

names:
  0: rip_current
  1: swimmer
"""
    yaml_path = yolo_dir / "data.yaml"
    with open(yaml_path, "w") as f:
        f.write(yaml_content)
    logger.info(f"Saved YOLO data.yaml to: {yaml_path}")


def main():
    parser = argparse.ArgumentParser(description="Prepare annotation files for RipVIS V2")
    parser.add_argument("--base_dir", type=str, default="data/V2",
                        help="Base directory for V2 dataset")
    args = parser.parse_args()

    setup_annotation_structure(Path(args.base_dir))


if __name__ == "__main__":
    main()
