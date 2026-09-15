"""
blend_utils.py
──────────────
Image compositing and blending utilities for the imprinting pipeline.

Supports:
  - Alpha compositing with feathered edges
  - Poisson seamless blending (via OpenCV)
  - Gaussian feather blending
  - Geometric augmentations for rip current regions
"""

import numpy as np
import cv2
from PIL import Image
from typing import Tuple, Optional, Union
from pathlib import Path


# ──────────────────────────────────────────────────────────────────────────────
# Core blending operations
# ──────────────────────────────────────────────────────────────────────────────

def alpha_composite(
    background: np.ndarray,
    foreground: np.ndarray,
    mask: np.ndarray,
    feather_radius: int = 15
) -> np.ndarray:
    """
    Alpha-composite a foreground rip current region onto a background beach image.

    Args:
        background:     (H, W, 3) uint8 RGB background image.
        foreground:     (H, W, 3) uint8 RGB source image (contains the rip current).
        mask:           (H, W)    uint8 binary mask (255 = rip current pixels).
        feather_radius: Gaussian blur radius for edge softening.

    Returns:
        Composited (H, W, 3) uint8 image.
    """
    assert background.shape == foreground.shape, \
        f"Shape mismatch: bg={background.shape}, fg={foreground.shape}"

    # Build a soft alpha from the binary mask
    if feather_radius > 0:
        alpha = cv2.GaussianBlur(mask.astype(np.float32) / 255.0,
                                  (feather_radius | 1, feather_radius | 1), 0)
    else:
        alpha = mask.astype(np.float32) / 255.0

    alpha = alpha[:, :, np.newaxis]                         # (H, W, 1)
    bg = background.astype(np.float32)
    fg = foreground.astype(np.float32)

    composite = fg * alpha + bg * (1.0 - alpha)
    return np.clip(composite, 0, 255).astype(np.uint8)


def poisson_blend(
    background: np.ndarray,
    foreground: np.ndarray,
    mask: np.ndarray,
    center: Optional[Tuple[int, int]] = None
) -> np.ndarray:
    """
    Poisson seamless blending using OpenCV seamlessClone.

    Pastes the masked region from `foreground` into `background` at `center`
    using Poisson editing for seamless boundary transitions.

    Args:
        background: (H, W, 3) uint8 RGB destination.
        foreground: (H, W, 3) uint8 RGB source (same size as background).
        mask:       (H, W)    uint8 binary mask (255 = region to blend).
        center:     (x, y) paste center; defaults to the mask centroid.

    Returns:
        Blended (H, W, 3) uint8 image.
    """
    # Compute center from mask centroid if not provided
    if center is None:
        M = cv2.moments(mask)
        if M["m00"] == 0:
            return background.copy()
        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])
        center = (cx, cy)

    # OpenCV expects BGR
    bg_bgr = cv2.cvtColor(background, cv2.COLOR_RGB2BGR)
    fg_bgr = cv2.cvtColor(foreground, cv2.COLOR_RGB2BGR)

    # Clamp center to valid range (seamlessClone crashes near borders)
    h, w = background.shape[:2]
    cx = int(np.clip(center[0], 1, w - 2))
    cy = int(np.clip(center[1], 1, h - 2))

    try:
        blended_bgr = cv2.seamlessClone(
            fg_bgr, bg_bgr, mask, (cx, cy), cv2.NORMAL_CLONE
        )
        return cv2.cvtColor(blended_bgr, cv2.COLOR_BGR2RGB)
    except cv2.error:
        # Fallback to alpha blending if Poisson fails
        return alpha_composite(background, foreground, mask, feather_radius=15)


def gaussian_feather_blend(
    background: np.ndarray,
    foreground: np.ndarray,
    mask: np.ndarray,
    sigma: float = 20.0
) -> np.ndarray:
    """
    Blend using a Gaussian-smoothed distance transform for natural feathering.

    Args:
        background: (H, W, 3) uint8 RGB.
        foreground: (H, W, 3) uint8 RGB.
        mask:       (H, W)    uint8 binary mask.
        sigma:      Gaussian sigma controlling blend width.

    Returns:
        (H, W, 3) uint8 blended image.
    """
    # Distance transform from mask edge → smoother than simple GaussianBlur
    dist = cv2.distanceTransform(mask, cv2.DIST_L2, 5)
    if dist.max() > 0:
        dist_norm = dist / dist.max()
    else:
        dist_norm = dist

    # Apply Gaussian to make it smooth
    alpha = cv2.GaussianBlur(dist_norm, (0, 0), sigma)
    alpha = np.clip(alpha, 0, 1)[:, :, np.newaxis]

    composite = foreground.astype(np.float32) * alpha + \
                background.astype(np.float32) * (1.0 - alpha)
    return np.clip(composite, 0, 255).astype(np.uint8)


# ──────────────────────────────────────────────────────────────────────────────
# Geometric augmentations for imprinting
# ──────────────────────────────────────────────────────────────────────────────

def augment_rip_region(
    region: np.ndarray,
    mask: np.ndarray,
    scale_range: Tuple[float, float] = (0.7, 1.3),
    rotation_range: Tuple[float, float] = (-15.0, 15.0),
    horizontal_flip: bool = True,
    brightness_jitter: float = 0.2,
    contrast_jitter: float = 0.1,
    rng: Optional[np.random.Generator] = None
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Apply random geometric and photometric augmentations to a rip current region.

    Args:
        region:             (H, W, 3) uint8 cropped rip current RGB region.
        mask:               (H, W)    uint8 binary mask of same size.
        scale_range:        (min_scale, max_scale) for random scaling.
        rotation_range:     (min_deg, max_deg) for random rotation.
        horizontal_flip:    If True, randomly flip horizontally.
        brightness_jitter:  Max absolute brightness change (fraction).
        contrast_jitter:    Max contrast change (fraction).
        rng:                NumPy random generator.

    Returns:
        augmented_region: (H', W', 3) uint8
        augmented_mask:   (H', W')    uint8 binary
    """
    if rng is None:
        rng = np.random.default_rng()

    h, w = region.shape[:2]

    # ── Horizontal flip ──
    if horizontal_flip and rng.random() > 0.5:
        region = cv2.flip(region, 1)
        mask = cv2.flip(mask, 1)

    # ── Random scale ──
    scale = float(rng.uniform(*scale_range))
    new_w = max(8, int(w * scale))
    new_h = max(8, int(h * scale))
    region = cv2.resize(region, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    mask = cv2.resize(mask, (new_w, new_h), interpolation=cv2.INTER_NEAREST)

    # ── Random rotation ──
    angle = float(rng.uniform(*rotation_range))
    if abs(angle) > 0.5:
        center = (new_w // 2, new_h // 2)
        M_rot = cv2.getRotationMatrix2D(center, angle, 1.0)
        region = cv2.warpAffine(region, M_rot, (new_w, new_h),
                                borderMode=cv2.BORDER_REFLECT)
        mask = cv2.warpAffine(mask, M_rot, (new_w, new_h),
                              borderMode=cv2.BORDER_CONSTANT, borderValue=0)

    # ── Brightness/Contrast jitter ──
    region = _color_jitter(region, brightness_jitter, contrast_jitter, rng)

    # Re-binarize mask after interpolation
    mask = (mask > 127).astype(np.uint8) * 255

    return region, mask


def _color_jitter(
    image: np.ndarray,
    brightness: float,
    contrast: float,
    rng: np.random.Generator
) -> np.ndarray:
    """Apply random brightness and contrast jitter to an RGB image."""
    img = image.astype(np.float32)

    # Brightness
    b_delta = float(rng.uniform(-brightness, brightness)) * 255.0
    img += b_delta

    # Contrast
    c_factor = 1.0 + float(rng.uniform(-contrast, contrast))
    img = (img - 128.0) * c_factor + 128.0

    return np.clip(img, 0, 255).astype(np.uint8)


# ──────────────────────────────────────────────────────────────────────────────
# Paste rip region onto background
# ──────────────────────────────────────────────────────────────────────────────

def paste_region_onto_background(
    background: np.ndarray,
    region: np.ndarray,
    mask: np.ndarray,
    position: Optional[Tuple[int, int]] = None,
    blend_method: str = "poisson",
    feather_radius: int = 15,
    rng: Optional[np.random.Generator] = None
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Paste a (possibly augmented) rip current region onto a background image.

    Args:
        background:     (H, W, 3) uint8 RGB beach background.
        region:         (rh, rw, 3) uint8 RGB rip current region.
        mask:           (rh, rw)    uint8 binary mask.
        position:       (x, y) top-left corner for pasting; random if None.
        blend_method:   "poisson" | "alpha" | "gaussian_feather"
        feather_radius: Feather radius for alpha blending.
        rng:            NumPy random generator.

    Returns:
        composite:     (H, W, 3) uint8 — blended result
        full_mask:     (H, W)    uint8 — mask placed at the same location
    """
    if rng is None:
        rng = np.random.default_rng()

    H, W = background.shape[:2]
    rh, rw = region.shape[:2]

    # Resize region if larger than background
    if rh > H or rw > W:
        scale = min(H / rh, W / rw) * 0.9
        new_rw = max(8, int(rw * scale))
        new_rh = max(8, int(rh * scale))
        region = cv2.resize(region, (new_rw, new_rh), interpolation=cv2.INTER_LINEAR)
        mask = cv2.resize(mask, (new_rw, new_rh), interpolation=cv2.INTER_NEAREST)
        rh, rw = new_rh, new_rw
        mask = (mask > 127).astype(np.uint8) * 255

    # Random position if not specified
    if position is None:
        x = int(rng.integers(0, max(1, W - rw)))
        y = int(rng.integers(0, max(1, H - rh)))
    else:
        x, y = position
        x = int(np.clip(x, 0, W - rw))
        y = int(np.clip(y, 0, H - rh))

    # Build full-size foreground and mask
    full_fg = background.copy()
    full_fg[y:y + rh, x:x + rw] = region
    full_mask = np.zeros((H, W), dtype=np.uint8)
    full_mask[y:y + rh, x:x + rw] = mask

    # Blend
    if blend_method == "poisson":
        composite = poisson_blend(background, full_fg, full_mask)
    elif blend_method == "gaussian_feather":
        composite = gaussian_feather_blend(background, full_fg, full_mask)
    else:  # alpha
        composite = alpha_composite(background, full_fg, full_mask, feather_radius)

    return composite, full_mask


# ──────────────────────────────────────────────────────────────────────────────
# Visualization helpers
# ──────────────────────────────────────────────────────────────────────────────

def overlay_mask_on_image(
    image: np.ndarray,
    mask: np.ndarray,
    color: Tuple[int, int, int] = (255, 60, 60),
    alpha: float = 0.45
) -> np.ndarray:
    """
    Overlay a semi-transparent colored mask on an RGB image for visualization.

    Args:
        image:  (H, W, 3) uint8 RGB image.
        mask:   (H, W)    uint8 binary mask.
        color:  RGB color tuple for the overlay.
        alpha:  Opacity of the overlay (0=transparent, 1=opaque).

    Returns:
        Visualized image (H, W, 3) uint8.
    """
    overlay = image.copy().astype(np.float32)
    binary = (mask > 0)
    for c, col in enumerate(color):
        overlay[binary, c] = overlay[binary, c] * (1 - alpha) + col * alpha
    return np.clip(overlay, 0, 255).astype(np.uint8)


def make_comparison_grid(
    originals: list,
    synthetics: list,
    n_cols: int = 4,
    cell_size: Tuple[int, int] = (256, 256)
) -> np.ndarray:
    """
    Create a side-by-side grid comparing original and synthetic images.

    Args:
        originals:  List of (H, W, 3) uint8 images.
        synthetics: List of (H, W, 3) uint8 images (same length).
        n_cols:     Number of image pairs per row.
        cell_size:  (width, height) per cell.

    Returns:
        Grid image as (H_grid, W_grid, 3) uint8.
    """
    assert len(originals) == len(synthetics)
    cw, ch = cell_size
    n = len(originals)
    n_rows = (n + n_cols - 1) // n_cols

    grid = np.ones((n_rows * ch, n_cols * 2 * cw, 3), dtype=np.uint8) * 240

    for idx, (orig, synth) in enumerate(zip(originals, synthetics)):
        row = idx // n_cols
        col = idx % n_cols

        orig_resized  = cv2.resize(orig,  (cw, ch))
        synth_resized = cv2.resize(synth, (cw, ch))

        y0 = row * ch
        x0_orig  = col * 2 * cw
        x0_synth = x0_orig + cw

        grid[y0:y0 + ch, x0_orig:x0_orig + cw]   = orig_resized
        grid[y0:y0 + ch, x0_synth:x0_synth + cw] = synth_resized

    return grid


if __name__ == "__main__":
    print("Testing blend_utils...")
    rng = np.random.default_rng(42)

    # Create dummy images
    bg = np.random.randint(50, 200, (480, 640, 3), dtype=np.uint8)
    fg = np.random.randint(50, 200, (480, 640, 3), dtype=np.uint8)
    mask = np.zeros((480, 640), dtype=np.uint8)
    mask[100:300, 200:450] = 255

    # Test alpha composite
    result_alpha = alpha_composite(bg, fg, mask, feather_radius=15)
    print(f"  alpha_composite: shape={result_alpha.shape} ✓")

    # Test gaussian feather
    result_gf = gaussian_feather_blend(bg, fg, mask)
    print(f"  gaussian_feather_blend: shape={result_gf.shape} ✓")

    # Test augmentation
    region = fg[100:300, 200:450]
    region_mask = mask[100:300, 200:450]
    aug_region, aug_mask = augment_rip_region(region, region_mask, rng=rng)
    print(f"  augment_rip_region: {region.shape} → {aug_region.shape} ✓")

    # Test paste
    composite, full_mask = paste_region_onto_background(
        bg, aug_region, aug_mask, blend_method="alpha", rng=rng
    )
    print(f"  paste_region_onto_background: shape={composite.shape} ✓")

    print("  blend_utils: OK ✓")
