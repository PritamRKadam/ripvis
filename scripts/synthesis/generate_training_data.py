"""
generate_training_data.py
─────────────────────────
One-shot script that builds a full YOLO-ready training dataset from scratch:

1. Creates N procedural beach background images (no internet/dataset needed)
2. Runs the imprinting pipeline (copy-paste + blending)
3. Splits into train/val sets (80/20)
4. Converts COCO annotations → YOLO segmentation .txt labels
5. Writes data.yaml for Ultralytics training

Usage:
    python3 scripts/generate_training_data.py --n_images 500 --output data/yolo_dataset/
    python3 scripts/generate_training_data.py --n_images 100 --output data/yolo_dataset/ --quick
"""

import argparse
import sys
import shutil
import random
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
from PIL import Image
from tqdm import tqdm


def parse_args():
    p = argparse.ArgumentParser(
        description="Generate YOLO-ready synthetic rip current training dataset."
    )
    p.add_argument("--n_images",   type=int, default=500,
                   help="Total synthetic images to generate (default: 500)")
    p.add_argument("--output",     type=str, default="data/yolo_dataset",
                   help="YOLO dataset output directory (default: data/yolo_dataset/)")
    p.add_argument("--val_split",  type=float, default=0.2,
                   help="Validation fraction (default: 0.2)")
    p.add_argument("--n_aug",      type=int, default=5,
                   help="Augmentations per source image (default: 5)")
    p.add_argument("--blend",      type=str, default="alpha",
                   choices=["alpha", "poisson", "gaussian_feather"],
                   help="Blending method (default: alpha)")
    p.add_argument("--img_size",   type=int, nargs=2, default=[640, 640],
                   metavar=("W", "H"),
                   help="Output image size (default: 640 640)")
    p.add_argument("--seed",       type=int, default=42)
    p.add_argument("--quick",      action="store_true",
                   help="Quick test: generate only 20 images, 2 epochs")
    p.add_argument("--source_dir", type=str, default=None,
                   help="Use existing images+masks instead of generating demo data")
    return p.parse_args()


# ──────────────────────────────────────────────────────────────────────────────
# Step 1: Generate source data
# ──────────────────────────────────────────────────────────────────────────────

def create_source_data(n_sources: int, output_dir: Path, img_size: tuple,
                       rng: np.random.Generator) -> Path:
    """
    Generate procedural beach images with rip current masks.
    Returns the directory containing images/ and masks/.
    """
    from download_samples import create_demo_beach_images

    demo_dir = output_dir / "_source_data"
    if (demo_dir / "images").is_dir() and len(list((demo_dir / "images").glob("*.png"))) >= n_sources:
        print(f"  Using existing source data: {demo_dir} ({n_sources} images)")
        return demo_dir

    print(f"  Generating {n_sources} procedural source images...")
    create_demo_beach_images(demo_dir, n=n_sources)
    return demo_dir


# ──────────────────────────────────────────────────────────────────────────────
# Step 2: Run imprinting pipeline
# ──────────────────────────────────────────────────────────────────────────────

def generate_synthetic_images(
    source_dir: Path,
    output_dir: Path,
    n_aug: int,
    blend_method: str,
    img_size: tuple,
    rng: np.random.Generator
) -> dict:
    """Imprint rip currents onto backgrounds, return COCO annotation dict."""
    from imprint_rip import collect_source_pairs, run_imprinting_pipeline
    from utils.coco_utils import COCODatasetBuilder
    import cv2

    img_src  = source_dir / "images"
    mask_src = source_dir / "masks"
    pairs    = collect_source_pairs(img_src, mask_src)

    if not pairs:
        raise ValueError(f"No image+mask pairs found in {source_dir}")

    bgs = list(img_src.glob("*.png"))
    print(f"  Source pairs: {len(pairs)}, backgrounds: {len(bgs)}")
    print(f"  Expected output: ~{len(pairs) * n_aug} images")

    coco    = COCODatasetBuilder(dataset_name="RipVIS-YOLO-Training")
    raw_out = output_dir / "_raw_synthetic"

    stats = run_imprinting_pipeline(
        source_pairs=pairs,
        background_paths=bgs,
        output_dir=raw_out,
        n_augmentations=n_aug,
        blend_method=blend_method,
        min_mask_area=800,
        save_overlays=False,
        rng=rng,
        coco_builder=coco
    )

    return {"coco": coco, "raw_dir": raw_out, "stats": stats}


# ──────────────────────────────────────────────────────────────────────────────
# Step 3: COCO → YOLO conversion
# ──────────────────────────────────────────────────────────────────────────────

def coco_ann_to_yolo_line(ann: dict, img_w: int, img_h: int) -> str | None:
    """
    Convert one COCO polygon annotation to a YOLO segmentation label line.
    Returns None if the polygon is degenerate.

    YOLO format: <class_id> x1 y1 x2 y2 ... xn yn   (all normalized 0-1)
    """
    class_id = ann["category_id"] - 1  # YOLO is 0-indexed

    # Flatten all polygon points
    all_pts = []
    for seg in ann["segmentation"]:
        pts = np.array(seg, dtype=np.float32).reshape(-1, 2)
        all_pts.append(pts)

    if not all_pts:
        return None

    pts = np.concatenate(all_pts, axis=0)
    if len(pts) < 3:
        return None

    # Normalize
    pts[:, 0] = np.clip(pts[:, 0] / img_w, 0, 1)
    pts[:, 1] = np.clip(pts[:, 1] / img_h, 0, 1)

    coords = " ".join(f"{x:.6f} {y:.6f}" for x, y in pts)
    return f"{class_id} {coords}"


def convert_coco_to_yolo(
    coco_data: dict,
    raw_dir: Path,
    yolo_dir: Path,
    val_split: float = 0.2,
    seed: int = 42
) -> dict:
    """
    Split images into train/val, copy images, and write YOLO label .txt files.

    Returns split counts.
    """
    # Build image_id → (filename, w, h) map
    id_to_img = {
        img["id"]: img
        for img in coco_data["images"]
    }
    # Build image_id → [annotations] map
    from collections import defaultdict
    id_to_anns = defaultdict(list)
    for ann in coco_data["annotations"]:
        id_to_anns[ann["image_id"]].append(ann)

    # Filter: keep only images with at least one annotation
    valid_ids = [iid for iid, anns in id_to_anns.items() if len(anns) > 0]
    print(f"  Images with annotations: {len(valid_ids)} / {len(id_to_img)}")

    # Shuffle and split
    rng = np.random.default_rng(seed)
    rng.shuffle(valid_ids := np.array(valid_ids))
    n_val = max(1, int(len(valid_ids) * val_split))
    val_ids  = set(valid_ids[:n_val].tolist())
    train_ids = set(valid_ids[n_val:].tolist())

    # Create YOLO directory structure
    for split in ["train", "val"]:
        (yolo_dir / split / "images").mkdir(parents=True, exist_ok=True)
        (yolo_dir / split / "labels").mkdir(parents=True, exist_ok=True)

    counts = {"train": 0, "val": 0, "skipped": 0}

    for img_id in tqdm(valid_ids.tolist(), desc="Converting to YOLO"):
        img_info = id_to_img.get(img_id)
        if img_info is None:
            continue

        # Find source image file
        # img_info["file_name"] may be "imprinted/imp_000000.png" or just "imp_000000.png"
        fname = Path(img_info["file_name"]).name
        src_img = raw_dir / "images" / fname

        if not src_img.exists():
            # Try searching recursively
            found = list(raw_dir.rglob(fname))
            src_img = found[0] if found else None

        if src_img is None or not src_img.exists():
            counts["skipped"] += 1
            continue

        split = "val" if img_id in val_ids else "train"
        dst_img = yolo_dir / split / "images" / fname
        dst_lbl = yolo_dir / split / "labels" / (Path(fname).stem + ".txt")

        # Copy image
        shutil.copy2(src_img, dst_img)

        # Write YOLO label file
        w, h = img_info["width"], img_info["height"]
        lines = []
        for ann in id_to_anns[img_id]:
            line = coco_ann_to_yolo_line(ann, w, h)
            if line:
                lines.append(line)

        if not lines:
            counts["skipped"] += 1
            continue

        with open(dst_lbl, "w") as f:
            f.write("\n".join(lines) + "\n")

        counts[split] += 1

    return counts


# ──────────────────────────────────────────────────────────────────────────────
# Step 4: Write data.yaml
# ──────────────────────────────────────────────────────────────────────────────

def write_data_yaml(yolo_dir: Path) -> Path:
    """Write the Ultralytics data.yaml configuration file."""
    import yaml

    yaml_path = yolo_dir / "data.yaml"
    config = {
        "path": str(yolo_dir.resolve()),
        "train": "train/images",
        "val":   "val/images",
        "nc":    1,
        "names": ["rip_current"]
    }

    with open(yaml_path, "w") as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)

    print(f"  Wrote data.yaml → {yaml_path}")
    return yaml_path


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()

    if args.quick:
        args.n_images  = 20
        args.n_aug     = 2

    rng = np.random.default_rng(args.seed)
    output_dir = Path(args.output)
    img_size   = tuple(args.img_size)

    print("=" * 60)
    print("  RipVIS — YOLO Training Data Generator")
    print("=" * 60)
    print(f"  Target images:   {args.n_images}")
    print(f"  Output dir:      {output_dir}")
    print(f"  Blend method:    {args.blend}")
    print(f"  Image size:      {img_size}")
    print(f"  Val split:       {args.val_split*100:.0f}%")
    print()

    # ── Step 1: Source data ─────────────────────────────────────────────────
    print("[1/4] Preparing source data...")
    if args.source_dir:
        source_dir = Path(args.source_dir)
    else:
        n_sources = max(10, args.n_images // args.n_aug + 2)
        source_dir = create_source_data(n_sources, output_dir, img_size, rng)

    # ── Step 2: Generate synthetic images ───────────────────────────────────
    print(f"\n[2/4] Running imprinting pipeline (×{args.n_aug} augmentations)...")
    result = generate_synthetic_images(
        source_dir=source_dir,
        output_dir=output_dir,
        n_aug=args.n_aug,
        blend_method=args.blend,
        img_size=img_size,
        rng=rng
    )
    coco = result["coco"]
    raw_dir = result["raw_dir"]
    stats = result["stats"]
    print(f"  Generated: {stats['generated']} images, skipped: {stats['skipped']}")

    # ── Step 3: COCO → YOLO ─────────────────────────────────────────────────
    print("\n[3/4] Converting COCO annotations → YOLO format...")
    yolo_dir = output_dir
    counts = convert_coco_to_yolo(
        coco_data=coco._dataset,
        raw_dir=raw_dir,
        yolo_dir=yolo_dir,
        val_split=args.val_split,
        seed=args.seed
    )
    print(f"  Train: {counts['train']} images")
    print(f"  Val:   {counts['val']} images")
    print(f"  Skipped: {counts['skipped']}")

    # ── Step 4: data.yaml ───────────────────────────────────────────────────
    print("\n[4/4] Writing data.yaml...")
    yaml_path = write_data_yaml(yolo_dir)

    # Save COCO JSON alongside
    import json
    coco_path = output_dir / "coco_annotations.json"
    coco.save(coco_path)

    print()
    print("=" * 60)
    print("  ✓ Dataset ready!")
    print(f"  Train: {counts['train']} images → {yolo_dir}/train/")
    print(f"  Val:   {counts['val']} images  → {yolo_dir}/val/")
    print(f"  YAML:  {yaml_path}")
    print()
    print("  Next step:")
    print(f"  python3 scripts/train_yolo.py --data {yaml_path}")


if __name__ == "__main__":
    main()
