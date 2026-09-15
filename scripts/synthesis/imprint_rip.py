"""
imprint_rip.py
──────────────
Method A: Rip Current Imprinting (Copy-Paste + Poisson Blending)

Given a set of source image+mask pairs and a set of background beach images,
this script creates synthetic training data by:
  1. Cropping the rip current region from a source frame
  2. Applying geometric and photometric augmentations
  3. Poisson-blending it onto a different background beach image
  4. Generating the corresponding COCO annotation

Usage:
    # Using demo images (no dataset required)
    python scripts/imprint_rip.py --source_dir data/samples/ --output_dir data/synthetic/ --test

    # Full run
    python scripts/imprint_rip.py \
        --source_dir data/samples/ \
        --background_dir data/raw/ \
        --mask_dir data/masks/ \
        --output_dir data/synthetic/ \
        --n_augmentations 5 \
        --blend_method poisson
"""

import argparse
import sys
import json
import random
from pathlib import Path
from itertools import cycle

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
from PIL import Image
from tqdm import tqdm

from utils.mask_utils import (
    load_mask, save_mask, load_image_and_mask,
    get_mask_area, get_bounding_box, crop_to_mask
)
from utils.blend_utils import (
    augment_rip_region, paste_region_onto_background,
    overlay_mask_on_image
)
from utils.coco_utils import COCODatasetBuilder


# ──────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="Imprint rip currents onto beach backgrounds (copy-paste + blending)."
    )
    p.add_argument("--source_dir",     type=str, default="data/samples",
                   help="Directory with source images (with rip currents).")
    p.add_argument("--mask_dir",       type=str, default=None,
                   help="Directory with binary rip current masks (default: source_dir/masks/).")
    p.add_argument("--background_dir", type=str, default=None,
                   help="Directory with background beach images (default: same as source_dir).")
    p.add_argument("--output_dir",     type=str, default="data/synthetic/imprinted",
                   help="Output directory for synthetic images and masks.")
    p.add_argument("--n_augmentations", type=int, default=5,
                   help="Number of synthetic composites per source pair (default: 5).")
    p.add_argument("--blend_method",   type=str, default="poisson",
                   choices=["poisson", "alpha", "gaussian_feather"],
                   help="Blending method (default: poisson).")
    p.add_argument("--feather_radius", type=int, default=15,
                   help="Feather radius for alpha blending (default: 15).")
    p.add_argument("--min_mask_area",  type=int, default=500,
                   help="Skip masks smaller than this (pixels, default: 500).")
    p.add_argument("--target_size",    type=int, nargs=2, default=None,
                   metavar=("W", "H"),
                   help="Resize all outputs to this size.")
    p.add_argument("--save_overlays",  action="store_true",
                   help="Save mask overlay visualization images.")
    p.add_argument("--annotation",     type=str, default=None,
                   help="Path for COCO annotation JSON output.")
    p.add_argument("--seed",           type=int, default=42,
                   help="Random seed for reproducibility.")
    p.add_argument("--test",           action="store_true",
                   help="Run a quick smoke test with synthetic demo data.")
    p.add_argument("--max_sources",     type=int, default=None,
                   help="Maximum number of source pairs to process.")
    return p.parse_args()


# ──────────────────────────────────────────────────────────────────────────────
# Pipeline
# ──────────────────────────────────────────────────────────────────────────────

def collect_source_pairs(
    source_dir: Path,
    mask_dir: Path
) -> list:
    """
    Collect (image_path, mask_path) pairs from directories.
    Matches by filename stem.
    """
    exts = {".png", ".jpg", ".jpeg", ".bmp"}
    image_files = {f.stem: f for f in source_dir.iterdir()
                   if f.is_file() and f.suffix.lower() in exts}
    mask_files  = {f.stem: f for f in mask_dir.iterdir()
                   if f.is_file() and f.suffix.lower() in exts}

    pairs = []
    for stem, img_path in image_files.items():
        if stem in mask_files:
            pairs.append((img_path, mask_files[stem]))

    return pairs


def run_imprinting_pipeline(
    source_pairs: list,
    background_paths: list,
    output_dir: Path,
    n_augmentations: int = 5,
    blend_method: str = "poisson",
    feather_radius: int = 15,
    min_mask_area: int = 500,
    target_size=None,
    save_overlays: bool = False,
    rng: np.random.Generator = None,
    coco_builder: COCODatasetBuilder = None,
    aug_config: dict = None
) -> dict:
    """
    Core imprinting loop.

    For each source (image, mask) pair:
      - Crop the rip current region
      - Augment it N times
      - Paste onto N different backgrounds
      - Save result + mask
      - Register in COCO builder

    Returns stats dict.
    """
    if rng is None:
        rng = np.random.default_rng(42)

    images_out  = output_dir / "images"
    masks_out   = output_dir / "masks"
    overlay_out = output_dir / "overlays"
    images_out.mkdir(parents=True, exist_ok=True)
    masks_out.mkdir(parents=True, exist_ok=True)
    if save_overlays:
        overlay_out.mkdir(parents=True, exist_ok=True)

    bg_cycle = cycle(background_paths)
    stats = {"generated": 0, "skipped": 0}
    global_idx = 0

    for src_img_path, src_mask_path in tqdm(source_pairs, desc="Source pairs"):
        # Load source
        src_img, src_mask = load_image_and_mask(
            src_img_path, src_mask_path, target_size
        )

        if get_mask_area(src_mask) < min_mask_area:
            stats["skipped"] += 1
            continue

        # Crop rip region + padding
        rip_region, rip_mask, (crop_x, crop_y, crop_w, crop_h) = \
            crop_to_mask(src_img, src_mask, padding=20)

        for aug_idx in range(n_augmentations):
            # Load background
            bg_path = next(bg_cycle)
            bg_img  = np.array(Image.open(bg_path).convert("RGB"))
            if target_size:
                from cv2 import resize, INTER_LINEAR
                bg_img = resize(bg_img, target_size, interpolation=INTER_LINEAR)

            # Augment the rip region
            aug_kwargs = {}
            if aug_config:
                if "scale_range" in aug_config:
                    aug_kwargs["scale_range"] = tuple(aug_config["scale_range"])
                if "rotation_range" in aug_config:
                    aug_kwargs["rotation_range"] = tuple(aug_config["rotation_range"])
                if "horizontal_flip" in aug_config:
                    aug_kwargs["horizontal_flip"] = aug_config["horizontal_flip"]
                if "brightness_jitter" in aug_config:
                    aug_kwargs["brightness_jitter"] = aug_config["brightness_jitter"]
                if "contrast_jitter" in aug_config:
                    aug_kwargs["contrast_jitter"] = aug_config["contrast_jitter"]

            aug_region, aug_mask = augment_rip_region(
                rip_region.copy(), rip_mask.copy(), rng=rng, **aug_kwargs
            )

            # Paste onto background
            composite, full_mask = paste_region_onto_background(
                background=bg_img,
                region=aug_region,
                mask=aug_mask,
                blend_method=blend_method,
                feather_radius=feather_radius,
                rng=rng
            )

            if get_mask_area(full_mask) < min_mask_area:
                continue

            # Save
            out_name = f"imp_{global_idx:06d}.png"
            Image.fromarray(composite).save(images_out / out_name)
            Image.fromarray(full_mask).save(masks_out  / out_name)

            if save_overlays:
                overlay = overlay_mask_on_image(composite, full_mask)
                Image.fromarray(overlay).save(overlay_out / out_name)

            # Register in COCO
            if coco_builder is not None:
                img_id, ann_ids = coco_builder.add_image_with_mask(
                    image_path=images_out / out_name,
                    mask=full_mask,
                    file_name=f"imprinted/{out_name}",
                    min_area=min_mask_area
                )

            global_idx += 1
            stats["generated"] += 1

    return stats


# ──────────────────────────────────────────────────────────────────────────────
# Test mode
# ──────────────────────────────────────────────────────────────────────────────

def run_test(output_dir: Path):
    """
    Smoke test: generate a small set of synthetic images using procedural data.
    No dataset download required.
    """
    print("\n[TEST MODE] Generating procedural beach demo data...")

    # First, create procedural samples
    sys.path.insert(0, str(Path(__file__).parent))
    from download_samples import create_demo_beach_images

    demo_dir = output_dir / "test_demo"
    create_demo_beach_images(demo_dir, n=6)

    source_dir = demo_dir / "images"
    mask_dir   = demo_dir / "masks"

    pairs = collect_source_pairs(source_dir, mask_dir)
    backgrounds = list((demo_dir / "images").glob("*.png"))

    if not pairs:
        print("ERROR: No source pairs found!")
        return

    rng = np.random.default_rng(42)
    coco = COCODatasetBuilder(dataset_name="RipVIS-Synthetic-Test")

    stats = run_imprinting_pipeline(
        source_pairs=pairs,
        background_paths=backgrounds,
        output_dir=output_dir / "test_output",
        n_augmentations=2,
        blend_method="alpha",
        save_overlays=True,
        rng=rng,
        coco_builder=coco
    )

    ann_path = output_dir / "test_output" / "annotations.json"
    coco.save(ann_path)

    print(f"\n✓ Test passed!")
    print(f"  Generated: {stats['generated']} synthetic images")
    print(f"  Skipped:   {stats['skipped']}")
    print(f"  Outputs:   {output_dir / 'test_output'}/")
    print(f"  COCO JSON: {ann_path}")


# ──────────────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    rng = np.random.default_rng(args.seed)

    print("=" * 60)
    print("  RipVIS — Imprinting Pipeline")
    print("=" * 60)

    output_dir = Path(args.output_dir)

    if args.test:
        run_test(output_dir)
        return

    # Resolve directories
    source_dir = Path(args.source_dir)
    mask_dir   = Path(args.mask_dir) if args.mask_dir else source_dir / "masks"
    bg_dir     = Path(args.background_dir) if args.background_dir else source_dir / "images"

    # Collect source pairs
    img_src_dir = source_dir / "images" if (source_dir / "images").is_dir() else source_dir
    pairs = collect_source_pairs(img_src_dir, mask_dir)
    
    # Shuffle and limit source pairs
    random.seed(args.seed)
    random.shuffle(pairs)
    if args.max_sources is not None:
        pairs = pairs[:args.max_sources]
        
    print(f"  Source pairs selected: {len(pairs)}")

    if not pairs:
        print("ERROR: No matching image+mask pairs found.")
        print(f"  Looked in: {img_src_dir} and {mask_dir}")
        return

    # Collect backgrounds
    exts = {".png", ".jpg", ".jpeg"}
    bg_img_dir = bg_dir / "images" if (bg_dir / "images").is_dir() else bg_dir
    backgrounds = [f for f in bg_img_dir.rglob("*") if f.suffix.lower() in exts]
    if not backgrounds:
        backgrounds = [p for p, _ in pairs]  # Fall back to source images

    print(f"  Backgrounds available: {len(backgrounds)}")
    print(f"  Augmentations/source:  {args.n_augmentations}")
    print(f"  Blend method:          {args.blend_method}")
    print(f"  Expected output:       ~{len(pairs) * args.n_augmentations} images")
    print()

    # COCO builder
    coco = COCODatasetBuilder(dataset_name="RipVIS-Synthetic-Imprinted")

    target_size = tuple(args.target_size) if args.target_size else None

    stats = run_imprinting_pipeline(
        source_pairs=pairs,
        background_paths=backgrounds,
        output_dir=output_dir,
        n_augmentations=args.n_augmentations,
        blend_method=args.blend_method,
        feather_radius=args.feather_radius,
        min_mask_area=args.min_mask_area,
        target_size=target_size,
        save_overlays=args.save_overlays,
        rng=rng,
        coco_builder=coco
    )

    # Save annotations
    ann_path = Path(args.annotation) if args.annotation else \
               output_dir / "imprinted_annotations.json"
    coco.save(ann_path)

    print(f"\n✓ Imprinting complete!")
    print(f"  Generated: {stats['generated']} synthetic images")
    print(f"  Skipped:   {stats['skipped']} (small masks)")
    print(f"  Output:    {output_dir}/")
    print(f"  COCO JSON: {ann_path}")


if __name__ == "__main__":
    main()
