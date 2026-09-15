"""
mask_utils.py
─────────────
Utility functions for loading, saving, and processing binary segmentation masks.
Used by both the imprinting and diffusion pipelines.
"""

import numpy as np
import cv2
from pathlib import Path
from PIL import Image
from typing import Union, Tuple, List, Optional
import json


# ──────────────────────────────────────────────────────────────────────────────
# I/O helpers
# ──────────────────────────────────────────────────────────────────────────────

def load_mask(path: Union[str, Path], binary: bool = True) -> np.ndarray:
    """
    Load a segmentation mask from disk.

    Args:
        path:   Path to the mask image (PNG, JPG, or NPY).
        binary: If True, return a bool/uint8 array where True = rip current pixel.

    Returns:
        np.ndarray of shape (H, W), dtype uint8, values 0 or 255 (if binary).
    """
    path = Path(path)
    if path.suffix == ".npy":
        mask = np.load(str(path))
    else:
        mask = np.array(Image.open(path).convert("L"))

    if binary:
        mask = (mask > 127).astype(np.uint8) * 255
    return mask


def save_mask(mask: np.ndarray, path: Union[str, Path]) -> None:
    """Save a mask array to disk as a grayscale PNG."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(mask).save(str(path))


def load_image_and_mask(
    image_path: Union[str, Path],
    mask_path: Union[str, Path],
    target_size: Optional[Tuple[int, int]] = None
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load an RGB image and its corresponding binary mask, optionally resizing both.

    Returns:
        image: np.ndarray (H, W, 3) uint8 RGB
        mask:  np.ndarray (H, W)    uint8 binary (0/255)
    """
    image = np.array(Image.open(image_path).convert("RGB"))
    mask = load_mask(mask_path)

    if target_size is not None:
        w, h = target_size
        image = cv2.resize(image, (w, h), interpolation=cv2.INTER_LINEAR)
        mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)

    return image, mask


# ──────────────────────────────────────────────────────────────────────────────
# Mask geometry
# ──────────────────────────────────────────────────────────────────────────────

def get_bounding_box(mask: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
    """
    Return (x, y, w, h) bounding box of the non-zero region in mask.
    Returns None if the mask is empty.
    """
    coords = np.argwhere(mask > 0)
    if len(coords) == 0:
        return None
    y0, x0 = coords.min(axis=0)
    y1, x1 = coords.max(axis=0)
    return int(x0), int(y0), int(x1 - x0 + 1), int(y1 - y0 + 1)


def get_mask_area(mask: np.ndarray) -> int:
    """Return the number of positive (rip current) pixels."""
    return int((mask > 0).sum())


def crop_to_mask(
    image: np.ndarray,
    mask: np.ndarray,
    padding: int = 10
) -> Tuple[np.ndarray, np.ndarray, Tuple[int, int, int, int]]:
    """
    Crop both image and mask to the bounding box of the mask region.

    Returns:
        cropped_image, cropped_mask, (x, y, w, h)
    """
    bbox = get_bounding_box(mask)
    if bbox is None:
        return image, mask, (0, 0, mask.shape[1], mask.shape[0])

    x, y, w, h = bbox
    H, W = mask.shape[:2]

    x1 = max(0, x - padding)
    y1 = max(0, y - padding)
    x2 = min(W, x + w + padding)
    y2 = min(H, y + h + padding)

    return image[y1:y2, x1:x2], mask[y1:y2, x1:x2], (x1, y1, x2 - x1, y2 - y1)


def mask_to_contours(mask: np.ndarray) -> List[np.ndarray]:
    """Extract OpenCV contours from a binary mask."""
    contours, _ = cv2.findContours(
        mask.astype(np.uint8),
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )
    return list(contours)


def contours_to_polygons(contours: List[np.ndarray], min_points: int = 6) -> List[List[float]]:
    """
    Convert OpenCV contours to flat polygon lists [x1, y1, x2, y2, ...],
    as used in COCO annotation format.

    Args:
        contours:   List of OpenCV contours (each shape: [N, 1, 2]).
        min_points: Minimum polygon vertices required (skip tiny contours).

    Returns:
        List of flat coordinate lists.
    """
    polygons = []
    for cnt in contours:
        cnt = cnt.squeeze(axis=1)  # (N, 2)
        if len(cnt) < min_points:
            continue
        flat = cnt.flatten().tolist()
        polygons.append(flat)
    return polygons


# ──────────────────────────────────────────────────────────────────────────────
# Morphological operations
# ──────────────────────────────────────────────────────────────────────────────

def dilate_mask(mask: np.ndarray, kernel_size: int = 5) -> np.ndarray:
    """Dilate mask to slightly expand the rip current region."""
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    return cv2.dilate(mask, kernel, iterations=1)


def erode_mask(mask: np.ndarray, kernel_size: int = 5) -> np.ndarray:
    """Erode mask to shrink the rip current region."""
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    return cv2.erode(mask, kernel, iterations=1)


def smooth_mask_edges(mask: np.ndarray, blur_radius: int = 11) -> np.ndarray:
    """
    Apply Gaussian blur to mask edges for feathered compositing.
    Returns a float32 mask in [0, 1].
    """
    smoothed = cv2.GaussianBlur(mask.astype(np.float32) / 255.0,
                                 (blur_radius, blur_radius), 0)
    return smoothed


def generate_random_ellipse_mask(
    h: int,
    w: int,
    n_ellipses: int = 3,
    rng: Optional[np.random.Generator] = None
) -> np.ndarray:
    """
    Generate a random organic-looking mask using overlapping ellipses.
    Useful for creating synthetic rip current shape priors.

    Returns:
        Binary mask (H, W) uint8 with values 0/255.
    """
    if rng is None:
        rng = np.random.default_rng()

    mask = np.zeros((h, w), dtype=np.uint8)

    # Rip currents are typically narrow and elongated
    for _ in range(n_ellipses):
        cx = int(rng.integers(w // 4, 3 * w // 4))
        cy = int(rng.integers(h // 4, 3 * h // 4))
        rx = int(rng.integers(20, w // 6))   # narrow x-axis
        ry = int(rng.integers(50, h // 3))   # elongated y-axis
        angle = float(rng.integers(-30, 30))
        cv2.ellipse(mask, (cx, cy), (rx, ry), angle, 0, 360, 255, -1)

    return mask


# ──────────────────────────────────────────────────────────────────────────────
# COCO RLE helpers
# ──────────────────────────────────────────────────────────────────────────────

def mask_to_rle(mask: np.ndarray) -> dict:
    """
    Encode a binary mask to COCO RLE format using pycocotools.
    Falls back to polygon if pycocotools is unavailable.
    """
    try:
        from pycocotools import mask as coco_mask_util
        rle = coco_mask_util.encode(np.asfortranarray(mask.astype(np.uint8)))
        rle["counts"] = rle["counts"].decode("utf-8")
        return rle
    except ImportError:
        # Fallback: return polygon instead
        contours = mask_to_contours(mask)
        polygons = contours_to_polygons(contours)
        return {"type": "polygon", "polygons": polygons}


if __name__ == "__main__":
    # Quick smoke test
    import tempfile, os

    print("Testing mask_utils...")
    rng = np.random.default_rng(42)

    # Create random mask
    mask = generate_random_ellipse_mask(480, 640, n_ellipses=2, rng=rng)
    print(f"  Generated mask: shape={mask.shape}, area={get_mask_area(mask)}")

    bbox = get_bounding_box(mask)
    print(f"  Bounding box: {bbox}")

    contours = mask_to_contours(mask)
    polygons = contours_to_polygons(contours)
    print(f"  Contours: {len(contours)}, polygons with >=6 pts: {len(polygons)}")

    smoothed = smooth_mask_edges(mask)
    print(f"  Smoothed edges: min={smoothed.min():.3f} max={smoothed.max():.3f}")

    print("  mask_utils: OK ✓")
