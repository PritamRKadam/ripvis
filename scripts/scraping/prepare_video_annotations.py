#!/usr/bin/env python3
"""
prepare_video_annotations.py
────────────────────────────
Prepares Video Instance Segmentation (VIS) and object tracking annotation structures.

Formats supported:
- YouTube-VIS / COCO-VIS JSON format (linking annotations across video frames by track_id)
- Label Studio Video Object Tracking task import format
- Frame sequence registry
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
logger = logging.getLogger("RipVIS_V2_VideoAnnotator")


def prepare_video_annotations(base_dir: Path):
    frame_base = base_dir / "frames"
    ann_dir = base_dir / "annotations"
    ann_dir.mkdir(parents=True, exist_ok=True)

    seq_dirs = sorted([d for d in frame_base.iterdir() if d.is_dir()])
    logger.info(f"Registering {len(seq_dirs)} video sequences into VIS annotation format...")

    vis_data = {
        "info": {
            "description": "RipVIS V2 Video Instance Segmentation Dataset (People in Water & Rip Currents)",
            "version": "2.0",
            "year": datetime.date.today().year,
            "date_created": datetime.date.today().isoformat()
        },
        "categories": [
            {"id": 0, "name": "rip_current", "supercategory": "hazard"},
            {"id": 1, "name": "swimmer", "supercategory": "person"}
        ],
        "videos": [],
        "annotations": []
    }

    label_studio_video_tasks = []

    for vid_id, sdir in enumerate(seq_dirs):
        frames = sorted(list(sdir.glob("*.jpg")) + list(sdir.glob("*.png")))
        if not frames:
            continue

        # Inspect first frame for dimensions
        with Image.open(frames[0]) as im:
            w, h = im.size

        file_names = [str(f.relative_to(base_dir)) for f in frames]

        video_record = {
            "id": vid_id,
            "name": sdir.name,
            "width": w,
            "height": h,
            "length": len(frames),
            "file_names": file_names
        }
        vis_data["videos"].append(video_record)

        label_studio_video_tasks.append({
            "data": {
                "video_id": sdir.name,
                "frame_count": len(frames),
                "resolution": f"{w}x{h}",
                "first_frame_preview": f"/data/local-files/?d=V2/videos/frames/{sdir.name}/{frames[0].name}",
                "frame_directory": f"/data/local-files/?d=V2/videos/frames/{sdir.name}/"
            }
        })

    # Save VIS COCO template
    vis_path = ann_dir / "video_coco_v2_template.json"
    with open(vis_path, "w") as f:
        json.dump(vis_data, f, indent=2)
    logger.info(f"Saved Video Instance Segmentation (VIS) template to: {vis_path}")

    # Save Label Studio Video Tasks
    ls_path = ann_dir / "label_studio_video_tasks.json"
    with open(ls_path, "w") as f:
        json.dump(label_studio_video_tasks, f, indent=2)
    logger.info(f"Saved Label Studio video tracking task manifest to: {ls_path}")


def main():
    parser = argparse.ArgumentParser(description="Prepare VIS tracking annotations for RipVIS V2")
    parser.add_argument("--base_dir", type=str, default="data/V2/videos",
                        help="Base directory for V2 video dataset")
    args = parser.parse_args()

    prepare_video_annotations(Path(args.base_dir))


if __name__ == "__main__":
    main()
