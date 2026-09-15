#!/usr/bin/env python3
"""
extract_video_frames.py
───────────────────────
Extracts non-redundant temporal frame sequences from scraped video clips
for Video Instance Segmentation (VIS) and tracking.

Features:
- Configurable sampling rate (default: 2 fps) to capture stroke cycles
- Saves frames to data/V2/videos/frames/<video_id>/frame_0000.jpg
- Generates 4x4 keyframe contact sheets for inspection in data/V2/videos/previews/
- Generates animated WebP / GIF previews
- Builds a frame manifest linking frames to video sequences
"""

import argparse
import glob
import json
import logging
import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np
from PIL import Image
from tqdm import tqdm

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger("RipVIS_V2_FrameExtractor")


def make_keyframe_grid(
    frame_paths: List[Path],
    output_path: Path,
    num_samples: int = 16,
    cell_w: int = 320,
    cell_h: int = 180
):
    """
    Creates a 4x4 contact sheet from uniformly sampled keyframes across the video.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not frame_paths:
        return

    # Uniformly pick up to num_samples indices
    total = len(frame_paths)
    if total <= num_samples:
        selected = frame_paths
    else:
        indices = np.linspace(0, total - 1, num_samples, dtype=int)
        selected = [frame_paths[i] for i in indices]

    rows, cols = 4, 4
    grid = np.zeros((rows * cell_h, cols * cell_w, 3), dtype=np.uint8)

    for idx, p in enumerate(selected[:rows * cols]):
        r = idx // cols
        c = idx % cols
        try:
            im = cv2.imread(str(p))
            if im is not None:
                im_resized = cv2.resize(im, (cell_w, cell_h), interpolation=cv2.INTER_AREA)
                # Draw frame index
                cv2.putText(im_resized, f"#{idx+1}", (10, 24),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                grid[r * cell_h:(r + 1) * cell_h, c * cell_w:(c + 1) * cell_w] = im_resized
        except Exception:
            pass

    cv2.imwrite(str(output_path), grid, [cv2.IMWRITE_JPEG_QUALITY, 92])
    logger.info(f"  Exported keyframe grid: {output_path.name}")


def export_animated_preview(frame_paths: List[Path], output_path: Path, max_frames: int = 24, fps: int = 4):
    """
    Exports a lightweight animated WebP preview of the extracted sequence.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not frame_paths:
        return

    total = len(frame_paths)
    step = max(1, total // max_frames)
    selected = frame_paths[::step][:max_frames]

    pil_frames = []
    for p in selected:
        try:
            im = Image.open(p).convert("RGB")
            # Downscale for preview
            im.thumbnail((480, 270), Image.Resampling.BILINEAR)
            pil_frames.append(im)
        except Exception:
            pass

    if pil_frames:
        duration_ms = int(1000 / fps)
        pil_frames[0].save(
            output_path,
            save_all=True,
            append_images=pil_frames[1:],
            duration=duration_ms,
            loop=0,
            quality=80
        )
        logger.info(f"  Exported animated preview: {output_path.name}")


def process_video(
    video_path: Path,
    output_base_dir: Path,
    target_sample_fps: float = 2.0,
    max_frames_per_video: int = 60
) -> Dict:
    """
    Extracts frames from a single video at `target_sample_fps`.
    """
    video_id = video_path.stem
    frame_dir = output_base_dir / "frames" / video_id
    preview_dir = output_base_dir / "previews"
    frame_dir.mkdir(parents=True, exist_ok=True)
    preview_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        logger.warning(f"Cannot open video: {video_path}")
        return {}

    orig_fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    if not orig_fps or orig_fps <= 0:
        orig_fps = 25.0

    frame_interval = max(1, int(round(orig_fps / target_sample_fps)))
    saved_paths = []
    frame_idx = 0
    extracted_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret or extracted_idx >= max_frames_per_video:
            break

        if frame_idx % frame_interval == 0:
            frame_filename = f"frame_{extracted_idx:04d}.jpg"
            dest_path = frame_dir / frame_filename
            cv2.imwrite(str(dest_path), frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
            saved_paths.append(dest_path)
            extracted_idx += 1

        frame_idx += 1

    cap.release()

    # Generate preview assets
    if saved_paths:
        grid_out = preview_dir / f"{video_id}_keyframes.jpg"
        make_keyframe_grid(saved_paths, grid_out)

        webp_out = preview_dir / f"{video_id}_preview.webp"
        export_animated_preview(saved_paths, webp_out)

    return {
        "video_id": video_id,
        "source_file": video_path.name,
        "native_resolution": f"{width}x{height}",
        "native_fps": round(orig_fps, 2),
        "sampled_fps": target_sample_fps,
        "extracted_frames_count": len(saved_paths),
        "frames_directory": str(frame_dir.relative_to(output_base_dir))
    }


def main():
    parser = argparse.ArgumentParser(description="Extract video frames for RipVIS V2")
    parser.add_argument("--base_dir", type=str, default="data/V2/videos",
                        help="Base directory for V2 video dataset")
    parser.add_argument("--sample_fps", type=float, default=2.0,
                        help="Sample rate in frames per second (default: 2.0)")
    parser.add_argument("--max_frames", type=int, default=60,
                        help="Maximum frames per video sequence")
    args = parser.parse_args()

    base_dir = Path(args.base_dir)
    raw_dir = base_dir / "raw"
    video_files = sorted(list(raw_dir.rglob("*.mp4")) + list(raw_dir.rglob("*.webm")) + list(raw_dir.rglob("*.ogv")))

    logger.info(f"Found {len(video_files)} video clips to process in {raw_dir}...")
    extraction_manifest = []

    for vpath in video_files:
        logger.info(f"Processing video: {vpath.name}...")
        info = process_video(
            video_path=vpath,
            output_base_dir=base_dir,
            target_sample_fps=args.sample_fps,
            max_frames_per_video=args.max_frames
        )
        if info:
            extraction_manifest.append(info)

    # Save extraction manifest
    meta_dir = base_dir / "metadata"
    meta_dir.mkdir(parents=True, exist_ok=True)
    manifest_out = meta_dir / "extracted_sequences_manifest.json"
    with open(manifest_out, "w") as f:
        json.dump(extraction_manifest, f, indent=2)

    logger.info("\n" + "=" * 50)
    logger.info("  RipVIS V2 Video Frame Extraction Complete")
    logger.info("=" * 50)
    logger.info(f"  Processed Videos : {len(extraction_manifest)}")
    logger.info(f"  Total Extracted  : {sum(m['extracted_frames_count'] for m in extraction_manifest)} frames")
    logger.info(f"  Manifest written : {manifest_out}")
    logger.info("=" * 50)


if __name__ == "__main__":
    main()
