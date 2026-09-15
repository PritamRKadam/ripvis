"""
RipVIS Synthetic Dataset — Explore Data

This notebook demonstrates:
1. Loading and visualizing real beach images from RipVIS
2. Inspecting binary rip current masks
3. Visualizing the imprinted (copy-paste) results
4. Visualizing the diffusion-generated results

Run all cells after generating data with:
  python scripts/build_dataset.py --test
"""

# ──────────────────────────────────────────────────────────────────────────────
# Cell 1: Imports and setup
# ──────────────────────────────────────────────────────────────────────────────
# (In Jupyter: run this cell first)

import sys
import os
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path("..").resolve()
sys.path.insert(0, str(PROJECT_ROOT))
os.chdir(PROJECT_ROOT)

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from PIL import Image

from utils.mask_utils import (
    load_mask, get_mask_area, get_bounding_box,
    mask_to_contours, contours_to_polygons,
    generate_random_ellipse_mask
)
from utils.blend_utils import overlay_mask_on_image, make_comparison_grid

print("Imports OK")
print(f"Working directory: {os.getcwd()}")

# ──────────────────────────────────────────────────────────────────────────────
# Cell 2: Load and visualize sample images + masks
# ──────────────────────────────────────────────────────────────────────────────

def find_image_mask_pairs(base_dir: Path, max_n: int = 8):
    """Find matching (image, mask) pairs."""
    img_dir  = base_dir / "images" if (base_dir / "images").is_dir() else base_dir
    mask_dir = base_dir / "masks"

    if not img_dir.is_dir():
        return []

    exts = {".png", ".jpg", ".jpeg"}
    pairs = []
    for img_path in sorted(img_dir.iterdir()):
        if img_path.suffix.lower() not in exts:
            continue
        mask_path = mask_dir / img_path.name if mask_dir.is_dir() else None
        pairs.append((img_path, mask_path))
        if len(pairs) >= max_n:
            break
    return pairs


def visualize_pairs(pairs, title="Dataset Samples"):
    """Plot a grid of image / mask / overlay triplets."""
    n = len(pairs)
    if n == 0:
        print("No pairs found to display.")
        return

    fig, axes = plt.subplots(n, 3, figsize=(12, 4 * n))
    if n == 1:
        axes = [axes]

    fig.suptitle(title, fontsize=14, y=1.01)

    for row_idx, (img_path, mask_path) in enumerate(pairs):
        img = np.array(Image.open(img_path).convert("RGB"))

        if mask_path and mask_path.exists():
            mask = load_mask(mask_path)
            area = get_mask_area(mask)
        else:
            mask = generate_random_ellipse_mask(*img.shape[:2], n_ellipses=2)
            area = get_mask_area(mask)

        overlay = overlay_mask_on_image(img, mask, color=(255, 50, 50), alpha=0.5)

        axes[row_idx][0].imshow(img)
        axes[row_idx][0].set_title(f"Image: {img_path.name}")
        axes[row_idx][0].axis("off")

        axes[row_idx][1].imshow(mask, cmap="gray")
        axes[row_idx][1].set_title(f"Mask (area={area:,} px)")
        axes[row_idx][1].axis("off")

        axes[row_idx][2].imshow(overlay)
        axes[row_idx][2].set_title("Overlay (rip = red)")
        axes[row_idx][2].axis("off")

    plt.tight_layout()
    plt.show()


# Load from demo or samples directory
for data_dir in [
    Path("data/test_demo"),
    Path("data/samples"),
    Path("data/raw"),
]:
    if data_dir.is_dir():
        pairs = find_image_mask_pairs(data_dir, max_n=4)
        if pairs:
            visualize_pairs(pairs, title=f"Samples from: {data_dir}")
            break

# ──────────────────────────────────────────────────────────────────────────────
# Cell 3: Mask statistics across dataset
# ──────────────────────────────────────────────────────────────────────────────

def compute_mask_statistics(mask_dir: Path):
    """Compute area distribution and count of rip current masks."""
    exts = {".png", ".jpg", ".jpeg"}
    mask_files = [f for f in mask_dir.glob("*") if f.suffix.lower() in exts]

    areas = []
    for mf in mask_files:
        mask = load_mask(mf)
        area = get_mask_area(mask)
        if area > 0:
            areas.append(area)

    if not areas:
        print(f"No non-empty masks found in {mask_dir}")
        return

    areas = np.array(areas)
    print(f"Mask statistics in: {mask_dir}")
    print(f"  Count:  {len(areas)} non-empty masks")
    print(f"  Min:    {areas.min():,} px")
    print(f"  Max:    {areas.max():,} px")
    print(f"  Mean:   {areas.mean():,.0f} px")
    print(f"  Median: {np.median(areas):,.0f} px")

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].hist(areas, bins=20, color="steelblue", edgecolor="white")
    axes[0].set_xlabel("Mask area (pixels)")
    axes[0].set_ylabel("Count")
    axes[0].set_title("Rip Current Mask Area Distribution")

    axes[1].boxplot(areas, vert=True, patch_artist=True,
                    boxprops=dict(facecolor="steelblue", alpha=0.7))
    axes[1].set_ylabel("Mask area (pixels)")
    axes[1].set_title("Area Box Plot")

    plt.tight_layout()
    plt.show()


for mask_dir in [
    Path("data/test_demo/masks"),
    Path("data/samples/masks"),
    Path("data/masks"),
]:
    if mask_dir.is_dir():
        compute_mask_statistics(mask_dir)
        break

# ──────────────────────────────────────────────────────────────────────────────
# Cell 4: Compare real vs synthetic images
# ──────────────────────────────────────────────────────────────────────────────

def compare_real_vs_synthetic(real_dir: Path, synth_dir: Path, n: int = 4):
    """Side-by-side grid: real vs synthetic."""
    exts = {".png", ".jpg", ".jpeg"}

    real_imgs  = sorted(p for p in real_dir.rglob("*")  if p.suffix.lower() in exts)[:n]
    synth_imgs = sorted(p for p in synth_dir.rglob("*") if p.suffix.lower() in exts)[:n]

    if not real_imgs or not synth_imgs:
        print("Not enough images to compare yet. Run the generation pipeline first.")
        return

    n = min(len(real_imgs), len(synth_imgs), n)
    originals  = [np.array(Image.open(p).convert("RGB")) for p in real_imgs[:n]]
    synthetics = [np.array(Image.open(p).convert("RGB")) for p in synth_imgs[:n]]

    grid = make_comparison_grid(originals, synthetics, n_cols=n, cell_size=(256, 256))

    fig, ax = plt.subplots(figsize=(16, 4))
    ax.imshow(grid)
    ax.set_title("Real (left) vs Synthetic (right) — Rip Current Images", fontsize=13)
    ax.axis("off")

    real_patch  = mpatches.Patch(color="royalblue", label="Real images")
    synth_patch = mpatches.Patch(color="salmon",    label="Synthetic images")
    ax.legend(handles=[real_patch, synth_patch], loc="upper right", fontsize=10)

    plt.tight_layout()
    plt.show()


real_source  = next((d for d in [
    Path("data/test_demo/images"),
    Path("data/samples/images"),
] if d.is_dir()), None)

synth_source = next((d for d in [
    Path("data/test_output/images"),
    Path("data/synthetic/imprinted/images"),
    Path("data/synthetic/diffusion/images"),
] if d.is_dir()), None)

if real_source and synth_source:
    compare_real_vs_synthetic(real_source, synth_source, n=4)
else:
    print("Run 'python scripts/build_dataset.py --test' first to generate data.")
    print(f"  Real source found:      {real_source}")
    print(f"  Synthetic source found: {synth_source}")

print("\nNotebook 01 complete!")
