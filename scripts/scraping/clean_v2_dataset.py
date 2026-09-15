#!/usr/bin/env python3
"""
clean_v2_dataset.py
───────────────────
Data cleaning, deduplication, and quality control pipeline for RipVIS V2.

Features:
- Validates RGB format and readable image streams
- Removes corrupt images
- Filters out low resolution (< 640px)
- Perceptual deduplication using color/pixel hashes
- Infographic & graphic pruning (detects flat artificial colors / warning signs)
- Organizes clean images into data/V2/cleaned/images/
- Generates 4x4 visual quality preview grids in data/V2/cleaned/preview_grids/
"""

import argparse
import hashlib
import json
import logging
import os
import sys
from pathlib import Path
from typing import Dict, List, Set, Tuple

import cv2
import numpy as np
from PIL import Image

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger("RipVIS_V2_Cleaner")


def is_graphic_or_infographic(img_bgr: np.ndarray) -> bool:
    """
    Detects if an image is an infographic, diagram, or warning sign
    rather than a real aerial photograph.
    Checks:
    - Highly saturated primary colors (red/yellow warning signs)
    - Low color variance / large flat color regions
    """
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    sat = hsv[:, :, 1]
    val = hsv[:, :, 2]

    # Extreme saturation with high brightness (typical of graphic text/signs)
    bright_saturated = ((sat > 180) & (val > 180)).mean()
    if bright_saturated > 0.35:
        return True

    # Check color gradient variance: drawings/diagrams have very sparse gradients
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
    if laplacian_var < 15.0:  # Flat image or solid color
        return True

    return False


def make_preview_grid(images: List[Path], output_path: Path, rows: int = 4, cols: int = 4, cell_size: int = 320):
    """
    Generates a 4x4 preview grid for visual inspection.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    grid = np.zeros((rows * cell_size, cols * cell_size, 3), dtype=np.uint8)

    sample_imgs = images[:rows * cols]
    for idx, p in enumerate(sample_imgs):
        r = idx // cols
        c = idx % cols
        try:
            im = Image.open(p).convert("RGB")
            im_resized = np.array(im.resize((cell_size, cell_size), Image.Resampling.BILINEAR))
            grid[r * cell_size:(r + 1) * cell_size, c * cell_size:(c + 1) * cell_size] = im_resized
        except Exception:
            pass

    Image.fromarray(grid).save(output_path, quality=90)
    logger.info(f"Saved inspection grid to: {output_path}")


def clean_v2_dataset(base_dir: Path, min_res: int = 600):
    raw_dir = base_dir / "raw"
    clean_dir = base_dir / "cleaned"
    clean_img_dir = clean_dir / "images"
    preview_dir = clean_dir / "preview_grids"

    clean_img_dir.mkdir(parents=True, exist_ok=True)
    preview_dir.mkdir(parents=True, exist_ok=True)

    seen_hashes: Set[str] = set()
    cleaned_count = 0
    skipped_corrupt = 0
    skipped_low_res = 0
    skipped_dup = 0
    skipped_graphic = 0

    category_cleaned: Dict[str, List[Path]] = {
        "rip_currents": [],
        "swimmers_ocean": [],
        "surf_coastal": []
    }

    subdirs = [d for d in raw_dir.iterdir() if d.is_dir()]
    logger.info(f"Scanning raw directories: {[d.name for d in subdirs]}...")

    for cat_dir in subdirs:
        cat_name = cat_dir.name
        img_files = sorted(list(cat_dir.glob("*.jpg")) + list(cat_dir.glob("*.jpeg")) + list(cat_dir.glob("*.png")))

        for p in img_files:
            # 1. Check validity
            try:
                with Image.open(p) as im:
                    w, h = im.size
                    im.verify()
                # Reopen for pixel read
                img_cv = cv2.imread(str(p))
                if img_cv is None:
                    skipped_corrupt += 1
                    continue
            except Exception:
                skipped_corrupt += 1
                continue

            # 2. Check resolution
            if w < min_res or h < min_res:
                skipped_low_res += 1
                continue

            # 3. Check duplicate (via perceptual difference hash)
            small = cv2.resize(cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY), (16, 16))
            diff = small[:, 1:] > small[:, :-1]
            dhash = hashlib.md5(diff.tobytes()).hexdigest()

            if dhash in seen_hashes:
                skipped_dup += 1
                continue
            seen_hashes.add(dhash)

            # 4. Check infographic / graphic
            if is_graphic_or_infographic(img_cv):
                skipped_graphic += 1
                continue

            # Copy to cleaned dataset
            dest_name = f"{cat_name}_{cleaned_count:05d}_{w}x{h}.jpg"
            dest_path = clean_img_dir / dest_name
            cv2.imwrite(str(dest_path), img_cv, [cv2.IMWRITE_JPEG_QUALITY, 95])

            if cat_name in category_cleaned:
                category_cleaned[cat_name].append(dest_path)
            else:
                category_cleaned.setdefault(cat_name, []).append(dest_path)

            cleaned_count += 1

    # Generate 4x4 preview grids for each category
    for cat_name, filepaths in category_cleaned.items():
        if filepaths:
            grid_out = preview_dir / f"preview_grid_{cat_name}.jpg"
            make_preview_grid(filepaths, grid_out)

    logger.info("\n" + "=" * 50)
    logger.info("  RipVIS V2 Dataset Cleaning Summary")
    logger.info("=" * 50)
    logger.info(f"  Valid Cleaned Aerial Images : {cleaned_count}")
    logger.info(f"  Skipped (Corrupt)           : {skipped_corrupt}")
    logger.info(f"  Skipped (Resolution < {min_res}p): {skipped_low_res}")
    logger.info(f"  Skipped (Duplicates)        : {skipped_dup}")
    logger.info(f"  Skipped (Graphics/Diagrams) : {skipped_graphic}")
    logger.info(f"  Clean images stored in      : {clean_img_dir}")
    logger.info("=" * 50)


def main():
    parser = argparse.ArgumentParser(description="Clean and validate RipVIS V2 scraped images")
    parser.add_argument("--base_dir", type=str, default="data/V2",
                        help="Base directory for V2 dataset")
    parser.add_argument("--min_res", type=int, default=600,
                        help="Minimum width and height in pixels")
    args = parser.parse_args()

    clean_v2_dataset(Path(args.base_dir), min_res=args.min_res)


if __name__ == "__main__":
    main()
