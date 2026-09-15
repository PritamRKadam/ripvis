"""
person_utils.py
────────────────
Utilities for person cutout loading, procedural cutout generation,
perspective scaling, RGBA compositing, and localized refinement mask creation
for Method 3 (Hybrid Person Imprinting + Diffusion Refinement).
"""

import os
from pathlib import Path
from typing import List, Tuple, Dict, Optional, Union
import numpy as np
import cv2
from PIL import Image, ImageDraw


# ──────────────────────────────────────────────────────────────────────────────
# Procedural Cutout Generator (Zero-dependency fallback)
# ──────────────────────────────────────────────────────────────────────────────

def load_person_cutouts(cutouts_dir: Union[str, Path] = "data/person_cutouts") -> List[np.ndarray]:
    """
    Load real photographic RGBA aerial swimmer cutouts from directory.
    Excludes artificial/procedural drawing files.

    Returns:
        List of (H, W, 4) uint8 RGBA numpy arrays.
    """
    cutouts_dir = Path(cutouts_dir).resolve()
    if not cutouts_dir.exists():
        print(f"Directory '{cutouts_dir}' does not exist.")
        return []

    # Exclude procedural/animated drawings
    png_paths = [
        p for p in cutouts_dir.glob("*.png")
        if not p.name.startswith("cutout_")  # filter out old procedural drawings
    ]

    print(f"[person_utils] Loaded {len(png_paths)} real photographic swimmer PNG cutouts from '{cutouts_dir}'.")
    cutouts = []
    for path in png_paths:
        img = Image.open(path).convert("RGBA")
        cutouts.append(np.array(img))

    return cutouts


# ──────────────────────────────────────────────────────────────────────────────
# Photorealistic GrabCut Swimmer Extractor (Board-free & Zero Water Fringe)
# ──────────────────────────────────────────────────────────────────────────────

_CLEAN_CUTOUT_CACHE = {}

def extract_clean_swimmer_cutout(img_path: Union[str, Path], color_format: str = "BGR") -> np.ndarray:
    """
    Extracts an exact, board-free cutout of the swimmer body with zero foreign water fringe
    using iterative GrabCut alpha matting, HSV olive/green sea fringe filtering,
    and direct sunlight highlight compression.

    Args:
        img_path: Path to swimmer crop image.
        color_format: "BGR" (default for OpenCV) or "RGB" (for PIL).

    Returns:
        (H, W, 4) uint8 array [C1, C2, C3, Alpha].
    """
    key = (str(img_path), color_format)
    if key in _CLEAN_CUTOUT_CACHE:
        return _CLEAN_CUTOUT_CACHE[key].copy()

    img = cv2.imread(str(img_path))
    if img is None:
        raise FileNotFoundError(f"Cannot read image at {img_path}")
    h, w = img.shape[:2]

    mask = np.zeros((h, w), np.uint8)
    bgd_model = np.zeros((1, 65), np.float64)
    fgd_model = np.zeros((1, 65), np.float64)

    margin_x = max(5, int(w * 0.10))
    margin_y = max(5, int(h * 0.08))
    rect = (margin_x, margin_y, max(1, w - 2 * margin_x), max(1, h - 2 * margin_y))
    cv2.grabCut(img, mask, rect, bgd_model, fgd_model, 6, cv2.GC_INIT_WITH_RECT)

    body_mask = np.where((mask == 2) | (mask == 0), 0, 1).astype("uint8") * 255

    # Suppress foreign olive/green water fringe from AFO source images
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    olive_water = (hsv[:, :, 0] >= 28) & (hsv[:, :, 0] <= 88) & (hsv[:, :, 1] >= 35)
    body_mask[olive_water] = 0

    # Morphological refinement
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    body_mask = cv2.morphologyEx(body_mask, cv2.MORPH_OPEN, kernel)
    body_mask = cv2.morphologyEx(body_mask, cv2.MORPH_CLOSE, kernel)

    # Subpixel anti-aliased edge smoothing
    alpha = cv2.GaussianBlur(body_mask, (3, 3), 0.75)

    b, g, r = cv2.split(img)

    # Soften clipped solar specular glare so skin retains natural warm tone
    lum = 0.299 * r.astype(np.float32) + 0.587 * g.astype(np.float32) + 0.114 * b.astype(np.float32)
    clipped = lum > 215
    if np.any(clipped):
        r[clipped] = np.clip(r[clipped].astype(np.float32) * 0.88 + 18, 0, 255).astype(np.uint8)
        g[clipped] = np.clip(g[clipped].astype(np.float32) * 0.88 + 10, 0, 255).astype(np.uint8)
        b[clipped] = np.clip(b[clipped].astype(np.float32) * 0.82 + 5, 0, 255).astype(np.uint8)

    if color_format.upper() == "RGB":
        result = cv2.merge([r, g, b, alpha])
    else:
        result = cv2.merge([b, g, r, alpha])

    _CLEAN_CUTOUT_CACHE[key] = result
    return result.copy()


# ──────────────────────────────────────────────────────────────────────────────
# Physical Immersion & Beer-Lambert Depth Attenuation Blending
# ──────────────────────────────────────────────────────────────────────────────

def blend_swimmer_physically(
    canvas: np.ndarray,
    swimmer_rgba: np.ndarray,
    target_xy: Tuple[int, int],
    scale: float = 0.36,
    angle: float = 0.0,
    submerged_axis: str = "vertical",
    depth_intensity: float = 0.30,
    exposure_factor: float = 0.90,
    color_format: str = "BGR"
) -> Tuple[np.ndarray, np.ndarray, Tuple[int, int, int, int]]:
    """
    Blends swimmer onto ocean canvas with realistic water depth absorption,
    subsurface refraction, wavelength-dependent absorption (Beer-Lambert law),
    and zero halo/border artifacts.

    Args:
        canvas: (H, W, 3) uint8 canvas.
        swimmer_rgba: (H, W, 4) uint8 swimmer patch with alpha.
        target_xy: (tx, ty) pixel coordinates where swimmer center will be placed.
        scale: Rescale factor matching authentic drone perspective.
        angle: Rotation angle along rip/wave flow in degrees.
        submerged_axis: "vertical" (legs lower) or "horizontal" (body aligned with jet).
        depth_intensity: Physical attenuation strength (0.0 to 0.65).
        exposure_factor: Sun exposure multiplier.
        color_format: "BGR" (default) or "RGB".

    Returns:
        canvas: (H, W, 3) uint8 modified canvas.
        person_mask: (H, W) uint8 binary mask of placed swimmer.
        bbox: (x_min, y_min, width, height) of the swimmer.
    """
    h_bg, w_bg = canvas.shape[:2]
    h_sw, w_sw = swimmer_rgba.shape[:2]

    # Rescale to authentic drone perspective scale
    new_w = max(8, int(w_sw * scale))
    new_h = max(8, int(h_sw * scale))
    sw_scaled = cv2.resize(swimmer_rgba, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)

    # Rotate along flow direction
    if abs(angle) > 0.5:
        M = cv2.getRotationMatrix2D((new_w // 2, new_h // 2), angle, 1.0)
        sw_scaled = cv2.warpAffine(
            sw_scaled, M, (new_w, new_h),
            borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0, 0)
        )

    tx, ty = target_xy
    x1 = max(0, tx - new_w // 2)
    y1 = max(0, ty - new_h // 2)
    x2 = min(w_bg, x1 + new_w)
    y2 = min(h_bg, y1 + new_h)

    crop_w = x2 - x1
    crop_h = y2 - y1
    if crop_w <= 0 or crop_h <= 0:
        return canvas, np.zeros((h_bg, w_bg), dtype=np.uint8), (0, 0, 0, 0)

    sw_patch = sw_scaled[:crop_h, :crop_w]
    c_channels = sw_patch[:, :, :3].astype(np.float32) * exposure_factor
    alpha = (sw_patch[:, :, 3].astype(np.float32) / 255.0)[:, :, np.newaxis]

    roi = canvas[y1:y2, x1:x2].astype(np.float32)
    ambient_water = np.mean(roi, axis=(0, 1))

    # Compute physical depth gradient along body axis
    y_grid, x_grid = np.mgrid[0:crop_h, 0:crop_w]
    if submerged_axis == "vertical":
        depth_grad = (y_grid.astype(np.float32) / max(1, crop_h))
    else:
        depth_grad = (x_grid.astype(np.float32) / max(1, crop_w))

    depth_attenuation = np.clip(depth_grad * depth_intensity, 0.0, 0.65)[:, :, np.newaxis]

    # Subsurface water volume refraction on submerged limbs
    refracted = cv2.GaussianBlur(c_channels, (3, 3), 0.8)
    c_channels = c_channels * (1.0 - depth_attenuation * 0.6) + refracted * (depth_attenuation * 0.6)

    # Wavelength absorption (Beer-Lambert law):
    # Red absorbs fastest in seawater; Blue/Green penetrate deeper.
    if color_format.upper() == "BGR":
        # idx 0: Blue, idx 1: Green, idx 2: Red
        c_channels[:, :, 2] = c_channels[:, :, 2] * (1.0 - depth_attenuation[:, :, 0] * 1.25) + ambient_water[2] * (depth_attenuation[:, :, 0] * 1.25)
        c_channels[:, :, 1] = c_channels[:, :, 1] * (1.0 - depth_attenuation[:, :, 0] * 0.85) + ambient_water[1] * (depth_attenuation[:, :, 0] * 0.85)
        c_channels[:, :, 0] = c_channels[:, :, 0] * (1.0 - depth_attenuation[:, :, 0] * 0.55) + ambient_water[0] * (depth_attenuation[:, :, 0] * 0.55)
    else:
        # idx 0: Red, idx 1: Green, idx 2: Blue
        c_channels[:, :, 0] = c_channels[:, :, 0] * (1.0 - depth_attenuation[:, :, 0] * 1.25) + ambient_water[0] * (depth_attenuation[:, :, 0] * 1.25)
        c_channels[:, :, 1] = c_channels[:, :, 1] * (1.0 - depth_attenuation[:, :, 0] * 0.85) + ambient_water[1] * (depth_attenuation[:, :, 0] * 0.85)
        c_channels[:, :, 2] = c_channels[:, :, 2] * (1.0 - depth_attenuation[:, :, 0] * 0.55) + ambient_water[2] * (depth_attenuation[:, :, 0] * 0.55)

    # Clean alpha composite (zero outline)
    blended = c_channels * alpha + roi * (1.0 - alpha)
    canvas[y1:y2, x1:x2] = np.clip(blended, 0, 255).astype(np.uint8)

    # Construct person mask and bounding box
    person_mask = np.zeros((h_bg, w_bg), dtype=np.uint8)
    person_mask[y1:y2, x1:x2] = (sw_patch[:, :, 3] > 30).astype(np.uint8) * 255
    bbox = (x1, y1, crop_w, crop_h)

    return canvas, person_mask, bbox


# ──────────────────────────────────────────────────────────────────────────────
# Placement & Perspective Scaling
# ──────────────────────────────────────────────────────────────────────────────

def calculate_perspective_scale(
    y_center: float,
    img_height: int,
    min_scale: float = 0.08,
    max_scale: float = 0.25,
    horizon_ratio: float = 0.35
) -> float:
    """
    Compute target person height fraction relative to image height based on Y position.
    Objects near the horizon (top of water) are smaller; objects near bottom are larger.

    Args:
        y_center: Y pixel position of person center.
        img_height: Total image height in pixels.
        min_scale: Relative scale near horizon (e.g. 0.08 of img height).
        max_scale: Relative scale near bottom of frame (e.g. 0.25 of img height).
        horizon_ratio: Estimated Y location of horizon (0.0=top, 0.35=upper third).

    Returns:
        Relative scale factor in [min_scale, max_scale].
    """
    norm_y = y_center / float(img_height)
    # Clip Y relative to horizon
    rel_y = np.clip((norm_y - horizon_ratio) / (1.0 - horizon_ratio + 1e-5), 0.0, 1.0)
    # Linear perspective scaling with quadratic ease for realistic depth
    scale_factor = min_scale + (max_scale - min_scale) * (rel_y ** 1.3)
    return float(scale_factor)


# ──────────────────────────────────────────────────────────────────────────────
# Matting & Relighting Harmonization
# ──────────────────────────────────────────────────────────────────────────────

def harmonize_color_and_lighting(
    fg_rgb: np.ndarray,
    bg_crop: np.ndarray,
    alpha: np.ndarray,
    harmonization_strength: float = 0.75
) -> np.ndarray:
    """
    Relight and harmonize color statistics of the cutout foreground (RGB)
    with the surrounding background ocean water in CIELAB color space.

    Args:
        fg_rgb: (H, W, 3) float32 RGB cutout foreground.
        bg_crop: (H, W, 3) float32 RGB background ocean crop.
        alpha: (H, W, 1) float32 alpha mask in [0, 1].
        harmonization_strength: Blend factor between original and harmonized (0 to 1).

    Returns:
        (H, W, 3) float32 harmonized RGB foreground.
    """
    if harmonization_strength <= 0:
        return fg_rgb

    fg_u8 = np.clip(fg_rgb, 0, 255).astype(np.uint8)
    bg_u8 = np.clip(bg_crop, 0, 255).astype(np.uint8)

    fg_lab = cv2.cvtColor(fg_u8, cv2.COLOR_RGB2LAB).astype(np.float32)
    bg_lab = cv2.cvtColor(bg_u8, cv2.COLOR_RGB2LAB).astype(np.float32)

    fg_mask = (alpha[:, :, 0] > 0.1)
    bg_mask = (alpha[:, :, 0] < 0.5)

    if not np.any(fg_mask) or not np.any(bg_mask):
        return fg_rgb

    fg_mean, fg_std = cv2.meanStdDev(fg_lab, mask=fg_mask.astype(np.uint8))
    bg_mean, bg_std = cv2.meanStdDev(bg_lab, mask=bg_mask.astype(np.uint8))

    fg_mean = fg_mean.reshape(1, 1, 3)
    fg_std = np.maximum(fg_std.reshape(1, 1, 3), 1e-5)
    bg_mean = bg_mean.reshape(1, 1, 3)
    bg_std = bg_std.reshape(1, 1, 3)

    harmonized_lab = (fg_lab - fg_mean) * (bg_std / fg_std) + bg_mean
    harmonized_lab = np.clip(harmonized_lab, 0, 255).astype(np.uint8)

    harmonized_rgb = cv2.cvtColor(harmonized_lab, cv2.COLOR_LAB2RGB).astype(np.float32)
    out_rgb = fg_rgb * (1.0 - harmonization_strength) + harmonized_rgb * harmonization_strength
    return out_rgb


def apply_deep_alpha_matting(
    alpha_raw: np.ndarray,
    blur_radius: int = 3
) -> np.ndarray:
    """
    Apply distance-guided alpha matting and bilateral edge softening to cutout alpha channels.

    Args:
        alpha_raw: (H, W, 1) float32 alpha in [0, 1].
        blur_radius: Radius for edge softening.

    Returns:
        (H, W, 1) float32 matted alpha channel in [0, 1].
    """
    alpha_u8 = (np.clip(alpha_raw[:, :, 0], 0, 1) * 255.0).astype(np.uint8)

    matted_u8 = cv2.bilateralFilter(alpha_u8, d=max(3, blur_radius * 2 + 1), sigmaColor=75, sigmaSpace=75)

    dist = cv2.distanceTransform((matted_u8 > 10).astype(np.uint8), cv2.DIST_L2, 5)
    if dist.max() > 0:
        dist_norm = np.clip(dist / (dist.max() * 0.4 + 1e-5), 0.0, 1.0)
        matted_alpha = (matted_u8.astype(np.float32) / 255.0) * dist_norm
    else:
        matted_alpha = matted_u8.astype(np.float32) / 255.0

    return np.clip(matted_alpha, 0.0, 1.0)[:, :, np.newaxis]


def place_person_cutout(
    background_img: np.ndarray,
    cutout_rgba: np.ndarray,
    center_pos: Tuple[int, int],
    target_height_px: int,
    rotation_deg: float = 0.0,
    enable_harmonization: bool = True,
    enable_deep_matting: bool = True,
    harmonization_strength: float = 0.75
) -> Tuple[np.ndarray, np.ndarray, Tuple[int, int, int, int]]:
    """
    Composites an RGBA cutout onto background image and updates person mask with optional
    relighting harmonization and deep alpha matting.

    Args:
        background_img: (H, W, 3) uint8 RGB background image.
        cutout_rgba: (H_c, W_c, 4) uint8 RGBA cutout image.
        center_pos: (cx, cy) pixel coordinates where cutout center will be placed.
        target_height_px: Desired cutout height in pixels.
        rotation_deg: Rotation angle in degrees.
        enable_harmonization: Align cutout lighting & color statistics with ocean water.
        enable_deep_matting: Apply soft distance-guided alpha matting.
        harmonization_strength: Strength factor for LAB relighting (0.0 to 1.0).

    Returns:
        composite_img: (H, W, 3) uint8 composite image.
        person_mask: (H, W) uint8 binary mask (255 where person is pasted).
        bbox: (x_min, y_min, width, height) of the placed person bounding box.
    """
    bg_h, bg_w = background_img.shape[:2]
    c_h, c_w = cutout_rgba.shape[:2]

    # Calculate scale factor to match target height
    scale = target_height_px / float(c_h)
    new_w = max(1, int(round(c_w * scale)))
    new_h = max(1, int(round(c_h * scale)))

    # Resize cutout
    resized_pil = Image.fromarray(cutout_rgba).resize((new_w, new_h), Image.Resampling.BILINEAR)

    # Optional rotation
    if abs(rotation_deg) > 0.5:
        resized_pil = resized_pil.rotate(rotation_deg, expand=True, resample=Image.Resampling.BILINEAR)
        new_w, new_h = resized_pil.size

    resized_cutout = np.array(resized_pil)

    # Calculate paste coordinates
    cx, cy = center_pos
    x_min = cx - new_w // 2
    y_min = cy - new_h // 2
    x_max = x_min + new_w
    y_max = y_min + new_h

    # Compute valid overlapping region between cutout and background
    bg_x1 = max(0, x_min)
    bg_y1 = max(0, y_min)
    bg_x2 = min(bg_w, x_max)
    bg_y2 = min(bg_h, y_max)

    fg_x1 = bg_x1 - x_min
    fg_y1 = bg_y1 - y_min
    fg_x2 = fg_x1 + (bg_x2 - bg_x1)
    fg_y2 = fg_y1 + (bg_y2 - bg_y1)

    if bg_x2 <= bg_x1 or bg_y2 <= bg_y1:
        # Out of bounds
        return background_img.copy(), np.zeros((bg_h, bg_w), dtype=np.uint8), (0, 0, 0, 0)

    # Extract regions
    fg_crop = resized_cutout[fg_y1:fg_y2, fg_x1:fg_x2]
    fg_rgb = fg_crop[:, :, :3].astype(np.float32)
    alpha = (fg_crop[:, :, 3].astype(np.float32) / 255.0)[:, :, np.newaxis]

    composite = background_img.copy()
    bg_crop = composite[bg_y1:bg_y2, bg_x1:bg_x2].astype(np.float32)

    # Apply deep alpha matting if requested
    if enable_deep_matting:
        alpha = apply_deep_alpha_matting(alpha)

    # Apply relighting harmonization if requested
    if enable_harmonization:
        fg_rgb = harmonize_color_and_lighting(
            fg_rgb, bg_crop, alpha, harmonization_strength=harmonization_strength
        )

    # Composite alpha blending
    blended_crop = fg_rgb * alpha + bg_crop * (1.0 - alpha)
    composite[bg_y1:bg_y2, bg_x1:bg_x2] = np.clip(blended_crop, 0, 255).astype(np.uint8)

    # Build binary person mask
    person_mask = np.zeros((bg_h, bg_w), dtype=np.uint8)
    binary_fg_alpha = (fg_crop[:, :, 3] > 30).astype(np.uint8) * 255
    person_mask[bg_y1:bg_y2, bg_x1:bg_x2] = binary_fg_alpha

    bbox = (bg_x1, bg_y1, bg_x2 - bg_x1, bg_y2 - bg_y1)
    return composite, person_mask, bbox


# ──────────────────────────────────────────────────────────────────────────────
# Local Refinement Mask Helper
# ──────────────────────────────────────────────────────────────────────────────

def create_refinement_mask(
    person_mask: np.ndarray,
    dilate_radius: int = 25
) -> np.ndarray:
    """
    Creates a dilated localized mask around the person to guide Stable Diffusion
    inpainting refinement (generating realistic splashes/wake around the person).

    Args:
        person_mask: (H, W) uint8 binary mask of the inserted person.
        dilate_radius: Expansion radius in pixels.

    Returns:
        (H, W) uint8 binary mask for SD inpainting.
    """
    if np.sum(person_mask) == 0:
        return np.zeros_like(person_mask)

    kernel_size = dilate_radius * 2 + 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    dilated = cv2.dilate(person_mask, kernel, iterations=1)
    # Gaussian blur edge transitions for smooth inpainting blending
    blurred = cv2.GaussianBlur(dilated, (15, 15), 0)
    return blurred
