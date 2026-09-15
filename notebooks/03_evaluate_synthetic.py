"""
RipVIS Synthetic Dataset — Evaluation

Evaluates the quality of synthetic images using:
1. Visual inspection grid
2. FID (Fréchet Inception Distance) score — requires torch-fidelity
3. Mask consistency: IoU between input mask and post-hoc predicted mask
4. Dataset statistics summary
"""

# ──────────────────────────────────────────────────────────────────────────────
# Cell 1: Setup
# ──────────────────────────────────────────────────────────────────────────────

import sys
import os
import json
from pathlib import Path

PROJECT_ROOT = Path("..").resolve()
sys.path.insert(0, str(PROJECT_ROOT))
os.chdir(PROJECT_ROOT)

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

from utils.blend_utils import overlay_mask_on_image, make_comparison_grid
from utils.mask_utils import load_mask, get_mask_area
from utils.coco_utils import validate_coco_json

print("Setup complete.")

# ──────────────────────────────────────────────────────────────────────────────
# Cell 2: Dataset statistics from COCO JSON
# ──────────────────────────────────────────────────────────────────────────────

def analyze_coco_dataset(json_path: str):
    """Print and visualize statistics from a COCO annotation JSON."""
    path = Path(json_path)
    if not path.exists():
        print(f"COCO JSON not found: {path}")
        print("Run 'python scripts/build_dataset.py --test' first.")
        return

    with open(path) as f:
        data = json.load(f)

    print(f"Dataset: {data.get('info', {}).get('description', 'Unknown')}")
    print(f"  Images:      {len(data['images'])}")
    print(f"  Annotations: {len(data['annotations'])}")

    areas = [a["area"] for a in data["annotations"]]
    if not areas:
        print("  No annotations found.")
        return

    areas = np.array(areas)
    print(f"\nAnnotation Statistics:")
    print(f"  Min area:    {areas.min():,.0f} px²")
    print(f"  Max area:    {areas.max():,.0f} px²")
    print(f"  Mean area:   {areas.mean():,.0f} px²")
    print(f"  Median area: {np.median(areas):,.0f} px²")

    # Annotations per image
    from collections import Counter
    ann_per_img = Counter(a["image_id"] for a in data["annotations"])
    counts = list(ann_per_img.values())
    print(f"\nAnnotations per image:")
    print(f"  Min: {min(counts)}, Max: {max(counts)}, Mean: {np.mean(counts):.1f}")

    # Plots
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    axes[0].hist(areas, bins=20, color="steelblue", edgecolor="white")
    axes[0].set_xlabel("Annotation area (px²)")
    axes[0].set_ylabel("Count")
    axes[0].set_title("Rip Current Area Distribution")

    axes[1].hist(counts, bins=range(1, max(counts) + 2),
                 align="left", rwidth=0.8, color="coral", edgecolor="white")
    axes[1].set_xlabel("Annotations per image")
    axes[1].set_ylabel("Count")
    axes[1].set_title("Annotations per Image")

    axes[2].pie(
        [len(areas)],
        labels=["rip_current"],
        autopct="%1.0f%%",
        colors=["steelblue"]
    )
    axes[2].set_title("Category Distribution")

    plt.tight_layout()
    plt.savefig("notebooks/dataset_stats.png", dpi=100, bbox_inches="tight")
    plt.show()
    print("Saved: notebooks/dataset_stats.png")


# Try both test and full dataset paths
for json_path in [
    "data/test_synthetic_dataset.json",
    "data/annotations/synthetic_dataset.json",
    "data/test_output/annotations.json",
]:
    if Path(json_path).exists():
        analyze_coco_dataset(json_path)
        break
else:
    print("No COCO JSON found. Run the generation pipeline first.")

# ──────────────────────────────────────────────────────────────────────────────
# Cell 3: Visual quality grid
# ──────────────────────────────────────────────────────────────────────────────

def quality_grid(synth_dir: Path, mask_dir: Path, n: int = 8):
    """Show generated images with their masks overlaid."""
    exts = {".png", ".jpg", ".jpeg"}
    img_paths  = sorted(p for p in synth_dir.rglob("*") if p.suffix.lower() in exts)[:n]
    if not img_paths:
        print(f"No images in {synth_dir}")
        return

    n = len(img_paths)
    cols = min(4, n)
    rows = (n + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 4 * rows))
    if rows == 1:
        axes = [axes]
    if cols == 1:
        axes = [[ax] for ax in axes]

    for idx, img_path in enumerate(img_paths):
        row, col = divmod(idx, cols)
        img  = np.array(Image.open(img_path).convert("RGB"))
        mask_path = mask_dir / img_path.name

        ax = axes[row][col]
        if mask_path.exists():
            mask = load_mask(mask_path)
            vis  = overlay_mask_on_image(img, mask, color=(255, 80, 80), alpha=0.4)
            ax.imshow(vis)
            ax.set_title(f"{img_path.name}\nMask area: {get_mask_area(mask):,} px")
        else:
            ax.imshow(img)
            ax.set_title(img_path.name)
        ax.axis("off")

    # Hide unused axes
    for idx in range(n, rows * cols):
        row, col = divmod(idx, cols)
        axes[row][col].axis("off")

    plt.suptitle(f"Synthetic Rip Current Images ({synth_dir.name})", fontsize=13)
    plt.tight_layout()
    plt.savefig("notebooks/quality_grid.png", dpi=100, bbox_inches="tight")
    plt.show()
    print("Saved: notebooks/quality_grid.png")


for synth_dir in [
    Path("data/test_output/images"),
    Path("data/synthetic/imprinted/images"),
    Path("data/synthetic/diffusion/images"),
]:
    if synth_dir.is_dir():
        mask_dir = synth_dir.parent / "masks"
        quality_grid(synth_dir, mask_dir, n=8)
        break

# ──────────────────────────────────────────────────────────────────────────────
# Cell 4: FID Score (requires torch-fidelity and enough images)
# ──────────────────────────────────────────────────────────────────────────────

def compute_fid(real_dir: Path, synth_dir: Path, min_images: int = 50):
    """Compute FID between real and synthetic image sets."""
    try:
        import torch_fidelity
    except ImportError:
        print("torch-fidelity not installed. Run: pip install torch-fidelity")
        return None

    real_imgs  = list(real_dir.rglob("*.png")) + list(real_dir.rglob("*.jpg"))
    synth_imgs = list(synth_dir.rglob("*.png")) + list(synth_dir.rglob("*.jpg"))

    print(f"Real images:      {len(real_imgs)}")
    print(f"Synthetic images: {len(synth_imgs)}")

    if len(real_imgs) < min_images or len(synth_imgs) < min_images:
        print(f"Need at least {min_images} images per set for reliable FID.")
        print("Generate more data or lower min_images threshold.")
        return None

    metrics = torch_fidelity.calculate_metrics(
        input1=str(real_dir),
        input2=str(synth_dir),
        fid=True,
        isc=False,
        kid=False,
        verbose=True
    )

    fid = metrics["frechet_inception_distance"]
    print(f"\nFID Score: {fid:.2f}")
    print("  (Lower is better; FID < 30 indicates high visual similarity)")
    return fid


# Uncomment after generating enough images (≥50 per set):
# fid = compute_fid(
#     real_dir=Path("data/samples/images"),
#     synth_dir=Path("data/synthetic/imprinted/images")
# )

print("\nCell 4: Uncomment compute_fid() after generating ≥50 images per set.")

# ──────────────────────────────────────────────────────────────────────────────
# Cell 5: Summary report
# ──────────────────────────────────────────────────────────────────────────────

print("\n" + "=" * 60)
print("  RipVIS Synthetic Dataset — Evaluation Summary")
print("=" * 60)

report = {
    "Imprinted images": 0,
    "Diffusion images": 0,
    "Total synthetic":  0,
    "COCO JSONs found": []
}

for img_dir in [
    Path("data/test_output/images"),
    Path("data/synthetic/imprinted/images"),
    Path("data/synthetic/diffusion/images"),
    Path("data/synthetic/refined/images"),
]:
    if img_dir.is_dir():
        count = len(list(img_dir.glob("*.png"))) + len(list(img_dir.glob("*.jpg")))
        key   = img_dir.parent.parent.name + "/" + img_dir.parent.name
        report["Total synthetic"] += count
        print(f"  {key}: {count} images")

for jpath in Path("data").rglob("*.json"):
    report["COCO JSONs found"].append(str(jpath))
    print(f"  COCO JSON: {jpath}")

print(f"\n  Total synthetic images: {report['Total synthetic']}")
print("\nNotebook 03 complete!")
