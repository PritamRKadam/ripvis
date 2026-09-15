"""
RipVIS Synthetic Dataset — Diffusion Pipeline Demo

Interactive demo showing how Stable Diffusion inpainting is used
to generate rip currents. Shows step-by-step results.
"""

# ──────────────────────────────────────────────────────────────────────────────
# Cell 1: Setup
# ──────────────────────────────────────────────────────────────────────────────

import sys
import os
from pathlib import Path

PROJECT_ROOT = Path("..").resolve()
sys.path.insert(0, str(PROJECT_ROOT))
os.chdir(PROJECT_ROOT)

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

from utils.mask_utils import (
    load_mask, generate_random_ellipse_mask, dilate_mask,
    smooth_mask_edges, get_mask_area
)
from utils.blend_utils import overlay_mask_on_image

print("Setup complete.")

# ──────────────────────────────────────────────────────────────────────────────
# Cell 2: Demonstrate mask generation strategies
# ──────────────────────────────────────────────────────────────────────────────

def show_mask_strategies(img_path: str = None):
    """Show different ways to generate rip current masks."""

    if img_path and Path(img_path).exists():
        base_img = np.array(Image.open(img_path).convert("RGB").resize((512, 512)))
    else:
        # Use procedural beach demo
        from scripts.download_samples import create_demo_beach_images
        import tempfile
        tmp = Path(tempfile.mkdtemp())
        create_demo_beach_images(tmp, n=1)
        base_img = np.array(Image.open(list((tmp / "images").glob("*.png"))[0]).resize((512, 512)))

    h, w = base_img.shape[:2]
    rng = np.random.default_rng(42)

    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    fig.suptitle("Rip Current Mask Generation Strategies", fontsize=14)

    strategies = [
        ("Random Ellipses (1)", generate_random_ellipse_mask(h, w, n_ellipses=1, rng=rng)),
        ("Random Ellipses (3)", generate_random_ellipse_mask(h, w, n_ellipses=3, rng=rng)),
        ("Dilated (k=15)",      dilate_mask(generate_random_ellipse_mask(h, w, rng=rng), 15)),
        ("Soft Edges",          None),  # will be smooth
    ]

    for col, (name, mask) in enumerate(strategies):
        if mask is None:
            base_mask = generate_random_ellipse_mask(h, w, rng=rng)
            smooth    = smooth_mask_edges(base_mask, blur_radius=31)
            axes[0][col].imshow(smooth, cmap="RdYlBu_r", vmin=0, vmax=1)
        else:
            axes[0][col].imshow(mask, cmap="gray")
        axes[0][col].set_title(name)
        axes[0][col].axis("off")

        # Overlay on image
        show_mask = (smooth * 255).astype(np.uint8) if mask is None else mask
        overlay = overlay_mask_on_image(base_img, show_mask, color=(255, 80, 80))
        axes[1][col].imshow(overlay)
        axes[1][col].set_title(f"Applied to image")
        axes[1][col].axis("off")

    plt.tight_layout()
    plt.savefig("notebooks/mask_strategies.png", dpi=100, bbox_inches="tight")
    plt.show()
    print("Saved: notebooks/mask_strategies.png")


show_mask_strategies()

# ──────────────────────────────────────────────────────────────────────────────
# Cell 3: Prompt engineering — show prompt variants
# ──────────────────────────────────────────────────────────────────────────────

from scripts.diffusion_generate import POSITIVE_PROMPT, NEGATIVE_PROMPT, PROMPT_VARIANTS

print("=" * 60)
print("POSITIVE PROMPT:")
print(POSITIVE_PROMPT)
print()
print("NEGATIVE PROMPT:")
print(NEGATIVE_PROMPT)
print()
print("PROMPT VARIANTS:")
for i, p in enumerate(PROMPT_VARIANTS):
    print(f"  [{i}] {p}")

# ──────────────────────────────────────────────────────────────────────────────
# Cell 4: Run a single inference (if GPU/model available)
# ──────────────────────────────────────────────────────────────────────────────

def run_single_inference(image_path: str = None, n_steps: int = 20):
    """
    Run one Stable Diffusion inference and visualize the result.
    Set n_steps=5 for a quick preview (lower quality).
    """
    from scripts.diffusion_generate import get_device, load_pipeline, generate_with_pipeline

    device = get_device("auto")
    print(f"Device: {device}")

    print("Loading model (this may take a few minutes first time)...")
    pipe = load_pipeline("stabilityai/stable-diffusion-2-inpainting", device)

    if pipe is None:
        print("Model not available. Skipping inference.")
        return

    # Load or generate demo image
    if image_path and Path(image_path).exists():
        pil_img = Image.open(image_path).convert("RGB")
    else:
        from scripts.download_samples import create_demo_beach_images
        import tempfile
        tmp = Path(tempfile.mkdtemp())
        create_demo_beach_images(tmp, n=1)
        pil_img = Image.open(list((tmp / "images").glob("*.png"))[0]).convert("RGB")

    # Generate mask
    h, w = 512, 512
    rng = np.random.default_rng(42)
    mask_np = generate_random_ellipse_mask(h, w, n_ellipses=2, rng=rng)
    mask_np = dilate_mask(mask_np, kernel_size=25)
    pil_mask = Image.fromarray(mask_np)

    print(f"Running inference ({n_steps} steps)...")
    result = generate_with_pipeline(
        pipe, pil_img, pil_mask,
        prompt=PROMPT_VARIANTS[0],
        negative_prompt=NEGATIVE_PROMPT,
        n_steps=n_steps,
        guidance_scale=7.5,
        strength=0.85,
        seed=42,
        img_size=(512, 512)
    )

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].imshow(pil_img.resize((512, 512)))
    axes[0].set_title("Input Beach Image")
    axes[0].axis("off")
    axes[1].imshow(pil_mask, cmap="gray")
    axes[1].set_title("Rip Current Mask")
    axes[1].axis("off")
    axes[2].imshow(result)
    axes[2].set_title(f"Generated (SD2-inpainting, {n_steps} steps)")
    axes[2].axis("off")

    plt.tight_layout()
    plt.savefig("notebooks/diffusion_result.png", dpi=100, bbox_inches="tight")
    plt.show()
    print("Saved: notebooks/diffusion_result.png")
    return result


# Uncomment to run (requires diffusers + model download):
# result = run_single_inference(n_steps=20)

print("\nNotebook 02 complete! Uncomment Cell 4 to run actual diffusion inference.")
