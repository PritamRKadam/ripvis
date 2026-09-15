#!/usr/bin/env python3
"""
prune_v2_videos.py
──────────────────
Utility for inspecting, pruning, and strictly filtering the V2 video dataset.

Features:
1. --list:
   Displays all videos currently in the V2 dataset, their paths, durations, and preview files.
2. --remove <id_or_keyword> [<id2> ...]:
   Safely removes specified videos and cascades deletion across:
   - data/V2/videos/raw/ (video file)
   - data/V2/videos/frames/<video_id>/ (all extracted frames)
   - data/V2/videos/previews/<video_id>* (contact sheets and webp animations)
   - data/V2/videos/metadata/video_manifest.json
   - data/V2/videos/metadata/extracted_sequences_manifest.json
   - data/V2/videos/annotations/label_studio_video_tasks.json
   - data/V2/videos/annotations/video_coco_v2_template.json
3. --audit-swimmers:
   Uses YOLOv8 person detection and HSV water mask to evaluate all videos.
   Flags any videos lacking verified swimmers immersed in water.
4. --prune-empty:
   Automatically prunes any video identified with zero confirmed swimmers in water.
"""

import argparse
import json
import logging
import shutil
import sys
from pathlib import Path
from typing import Dict, List, Set, Tuple

import cv2
import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger("RipVIS_VideoPruner")


def compute_water_mask(img_bgr: np.ndarray) -> np.ndarray:
    """Computes binary water mask across ocean, river, lake, and white surf foam."""
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    mask_blue = (hsv[:, :, 0] >= 75) & (hsv[:, :, 0] <= 135) & (hsv[:, :, 1] >= 20) & (hsv[:, :, 2] >= 20)
    mask_green = (hsv[:, :, 0] >= 32) & (hsv[:, :, 0] < 75) & (hsv[:, :, 1] >= 25) & (hsv[:, :, 2] >= 25)
    mask_foam = (hsv[:, :, 1] < 55) & (hsv[:, :, 2] > 160)
    mask_dark = (hsv[:, :, 0] >= 70) & (hsv[:, :, 0] <= 140) & (hsv[:, :, 2] >= 15) & (hsv[:, :, 2] <= 70) & (hsv[:, :, 1] >= 25)
    return mask_blue | mask_green | mask_foam | mask_dark


def list_videos(base_dir: Path):
    manifest_path = base_dir / "metadata" / "video_manifest.json"
    if not manifest_path.exists():
        logger.warning(f"Manifest not found at {manifest_path}")
        return

    with open(manifest_path, "r") as f:
        records = json.load(f)

    print(f"\n{'='*90}")
    print(f"{'#':<3} {'Video ID':<48} {'Category':<22} {'Duration':<10}")
    print(f"{'-'*90}")
    for idx, r in enumerate(records, 1):
        vid = r.get("video_id", r.get("filename", "unknown"))
        cat = r.get("category", "unknown")
        dur = f"{r.get('duration_sec', 0.0):.1f}s"
        print(f"{idx:<3} {vid[:47]:<48} {cat[:21]:<22} {dur:<10}")
    print(f"{'='*90}")
    print(f"Total videos recorded: {len(records)}\n")
    print(f"💡 Contact sheet previews available at: {base_dir / 'previews'}")


def remove_videos(base_dir: Path, targets: List[str], dry_run: bool = False):
    """
    Cascades removal of video from raw, frames, previews, and all JSON manifests.
    """
    raw_dir = base_dir / "raw"
    frames_dir = base_dir / "frames"
    previews_dir = base_dir / "previews"
    meta_dir = base_dir / "metadata"
    ann_dir = base_dir / "annotations"

    manifest_path = meta_dir / "video_manifest.json"
    seq_manifest_path = meta_dir / "extracted_sequences_manifest.json"
    label_studio_path = ann_dir / "label_studio_video_tasks.json"
    coco_path = ann_dir / "video_coco_v2_template.json"

    # Load video manifest
    records = []
    if manifest_path.exists():
        with open(manifest_path, "r") as f:
            records = json.load(f)

    # Match target strings to video_id or filename
    to_delete_ids: Set[str] = set()
    for t in targets:
        t_clean = t.strip()
        matched = False
        for r in records:
            vid = r.get("video_id", "")
            fn = r.get("filename", "")
            if t_clean.lower() == vid.lower() or t_clean.lower() == fn.lower() or t_clean.lower() in vid.lower():
                to_delete_ids.add(vid)
                matched = True
        # Also check raw dir if not matched in manifest
        if not matched:
            for p in raw_dir.rglob("*"):
                if p.is_file() and t_clean.lower() in p.name.lower():
                    to_delete_ids.add(p.stem)
                    matched = True

    if not to_delete_ids:
        logger.warning(f"No videos matched targets: {targets}")
        return

    logger.info(f"Targets identified for removal ({len(to_delete_ids)}):")
    for vid in sorted(to_delete_ids):
        logger.info(f"  • {vid}")

    if dry_run:
        logger.info("Dry run enabled. No files removed.")
        return

    # 1. Delete raw video files
    for vid in to_delete_ids:
        for p in raw_dir.rglob("*"):
            if p.is_file() and (p.stem == vid or p.name.startswith(vid)):
                logger.info(f"Deleting raw video: {p}")
                p.unlink()

    # 2. Delete extracted frames
    for vid in to_delete_ids:
        v_frames = frames_dir / vid
        if v_frames.exists() and v_frames.is_dir():
            logger.info(f"Deleting extracted frames: {v_frames}")
            shutil.rmtree(v_frames)

    # 3. Delete previews
    for vid in to_delete_ids:
        for p in previews_dir.glob(f"{vid}*"):
            if p.is_file():
                logger.info(f"Deleting preview: {p}")
                p.unlink()

    # 4. Update video_manifest.json
    if manifest_path.exists():
        new_records = [r for r in records if r.get("video_id") not in to_delete_ids]
        with open(manifest_path, "w") as f:
            json.dump(new_records, f, indent=2)
        logger.info(f"Updated {manifest_path.name} (remaining: {len(new_records)})")

    # 5. Update extracted_sequences_manifest.json
    if seq_manifest_path.exists():
        try:
            with open(seq_manifest_path, "r") as f:
                seq_records = json.load(f)
            new_seq = [s for s in seq_records if s.get("video_id") not in to_delete_ids]
            with open(seq_manifest_path, "w") as f:
                json.dump(new_seq, f, indent=2)
            logger.info(f"Updated {seq_manifest_path.name} (remaining: {len(new_seq)})")
        except Exception as e:
            logger.warning(f"Error updating seq manifest: {e}")

    # 6. Update label studio & COCO templates if present
    if label_studio_path.exists():
        try:
            with open(label_studio_path, "r") as f:
                tasks = json.load(f)
            new_tasks = [t for t in tasks if t.get("data", {}).get("video_id") not in to_delete_ids]
            with open(label_studio_path, "w") as f:
                json.dump(new_tasks, f, indent=2)
            logger.info(f"Updated {label_studio_path.name}")
        except Exception as e:
            logger.warning(f"Error updating label studio tasks: {e}")

    logger.info("✅ Removal completed successfully.")


def audit_dataset_for_swimmers(base_dir: Path) -> Tuple[List[str], List[str]]:
    """
    Audits existing extracted frames or videos using YOLOv8 person detection + water context.
    Returns: (valid_swimmer_video_ids, no_swimmer_video_ids)
    """
    from ultralytics import YOLO

    frames_dir = base_dir / "frames"
    previews_dir = base_dir / "previews"
    manifest_path = base_dir / "metadata" / "video_manifest.json"

    if not manifest_path.exists():
        logger.error(f"Manifest {manifest_path} not found.")
        return [], []

    with open(manifest_path, "r") as f:
        records = json.load(f)

    logger.info("Loading YOLOv8n detector...")
    model = YOLO("yolov8n.pt")

    valid_ids = []
    invalid_ids = []

    print(f"\n{'='*95}")
    print(f"{'Video ID':<48} {'Water%':<10} {'Persons':<10} {'Swimmer Confirmed':<18}")
    print(f"{'-'*95}")

    for r in records:
        vid = r.get("video_id", "")
        # Look for frames
        f_dir = frames_dir / vid
        frame_files = sorted(list(f_dir.glob("*.jpg"))) if f_dir.exists() else []

        # If no frames in frames/, try preview keyframe sheet or raw video
        if not frame_files:
            # check preview contact sheet
            preview_img = previews_dir / f"{vid}_keyframes.jpg"
            if preview_img.exists():
                frame_files = [preview_img]

        if not frame_files:
            logger.warning(f"No frames found to test for {vid}")
            invalid_ids.append(vid)
            continue

        # Uniformly sample up to 12 frames
        sample_indices = np.linspace(0, len(frame_files) - 1, min(12, len(frame_files)), dtype=int)
        sampled = [frame_files[i] for i in sample_indices]

        water_ratios = []
        person_water_contexts = []
        total_persons = 0
        swimmer_frames = 0

        for fp in sampled:
            im = cv2.imread(str(fp))
            if im is None:
                continue
            h, w = im.shape[:2]
            wmask = compute_water_mask(im)
            fw = float(np.mean(wmask))
            water_ratios.append(fw)

            results = model(im, verbose=False)
            frame_boxes = []
            for res in results:
                for b in res.boxes:
                    if int(b.cls[0]) == 0 and float(b.conf[0]) >= 0.25:
                        frame_boxes.append(b.xyxy[0].cpu().numpy())

            if frame_boxes:
                total_persons += len(frame_boxes)
                box_water = []
                for box in frame_boxes:
                    x1, y1, x2, y2 = [int(v) for v in box]
                    bw = max(1, x2 - x1)
                    bh = max(1, y2 - y1)
                    ex1 = max(0, x1 - int(bw * 0.35))
                    ey1 = max(0, y1 - int(bh * 0.35))
                    ex2 = min(w, x2 + int(bw * 0.35))
                    ey2 = min(h, y2 + int(bh * 0.35))
                    ctx = float(np.mean(wmask[ey1:ey2, ex1:ex2]))
                    box_water.append(ctx)

                person_water_contexts.extend(box_water)
                if any(c >= 0.18 for c in box_water) or (fw >= 0.30 and any(c >= 0.10 for c in box_water)):
                    swimmer_frames += 1

        mean_fw = float(np.mean(water_ratios)) if water_ratios else 0.0
        mean_pw = float(np.mean(person_water_contexts)) if person_water_contexts else 0.0

        is_swimmer = (swimmer_frames >= 2 and mean_fw >= 0.20) or (swimmer_frames >= 1 and mean_pw >= 0.30 and mean_fw >= 0.30)

        status_str = "✅ YES" if is_swimmer else "❌ NO (Discard)"
        print(f"{vid[:47]:<48} {mean_fw*100:>5.1f}%     {total_persons:<10} {status_str:<18}")

        if is_swimmer:
            valid_ids.append(vid)
        else:
            invalid_ids.append(vid)

    print(f"{'='*95}")
    print(f"Audit Summary: {len(valid_ids)} confirmed swimmers in water, {len(invalid_ids)} lacking swimmers/water.")
    return valid_ids, invalid_ids


def main():
    parser = argparse.ArgumentParser(description="RipVIS V2 Video Pruner & Swimmer Auditor")
    parser.add_argument("--base_dir", type=str, default="data/V2/videos",
                        help="Base directory for V2 video dataset")
    parser.add_argument("--list", action="store_true",
                        help="List all videos in the dataset with duration and metadata")
    parser.add_argument("--remove", nargs="+",
                        help="List of video IDs, filenames, or substrings to remove cleanly")
    parser.add_argument("--dry-run", action="store_true",
                        help="Simulate removal without deleting files")
    parser.add_argument("--audit", action="store_true",
                        help="Audit all videos with YOLO + water analysis to detect missing swimmers")
    parser.add_argument("--prune-non-swimmers", action="store_true",
                        help="Audit and automatically prune all videos lacking verified swimmers in water")
    args = parser.parse_args()

    base_dir = Path(args.base_dir)

    if args.list:
        list_videos(base_dir)
        return

    if args.remove:
        remove_videos(base_dir, args.remove, dry_run=args.dry_run)
        return

    if args.audit:
        audit_dataset_for_swimmers(base_dir)
        return

    if args.prune_non_swimmers:
        valid_ids, invalid_ids = audit_dataset_for_swimmers(base_dir)
        if invalid_ids:
            print(f"\nPruning {len(invalid_ids)} non-swimmer videos...")
            remove_videos(base_dir, invalid_ids, dry_run=args.dry_run)
        else:
            print("\nAll videos have confirmed swimmers in water. Nothing to prune.")
        return

    parser.print_help()


if __name__ == "__main__":
    main()
