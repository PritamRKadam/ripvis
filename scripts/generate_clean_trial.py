"""
generate_clean_trial.py
───────────────────────
Generates 5 ultra-realistic trial scenes with visible swimmers in/near rip currents:
- Image 1: Cleaned version of the approved swimmer scene with the blue block completely removed
  and seamless water/swimmer harmonization.
- Image 2: Two swimmers navigating the rip current neck between incoming breaking waves.
- Image 3: Swimmer swimming freestyle inside the active rip current trench.
- Image 4: Swimmers treading water near the outgoing rip plume and wave break zone.
- Image 5: Swimmer caught in the expanding rip current head offshore.

Outputs directly to data/trial/ and creates data/trial/trial_5_samples.jpg.
"""

import cv2
import numpy as np
from pathlib import Path
from PIL import Image, ImageDraw


def clean_sample_01(src_path: Path, dst_path: Path):
    """Clean the approved Image 1 by removing the blue rectangular artifact."""
    img = cv2.imread(str(src_path))
    if img is None:
        raise FileNotFoundError(f"Cannot open {src_path}")
    
    # Exact bounding box of the blue artifact is around x=175..245, y=205..275
    roi = img[190:290, 160:260].copy()
    b, g, r = roi[:, :, 0], roi[:, :, 1], roi[:, :, 2]
    blue_pixels = (b.astype(int) - r.astype(int) > 50) & (b > 120)

    mask = np.zeros((100, 100), dtype=np.uint8)
    mask[blue_pixels] = 255
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    mask_dilated = cv2.dilate(mask, kernel, iterations=3)

    full_mask = np.zeros(img.shape[:2], dtype=np.uint8)
    full_mask[190:290, 160:260] = mask_dilated

    # Inpaint the blue artifact
    inpainted = cv2.inpaint(img, full_mask, inpaintRadius=9, flags=cv2.INPAINT_TELEA)

    # Place a clean, board-free swimmer treading water right beside the splash
    swimmer_file = Path("data/afo_samples/a_1044_crops/46_human_135x162.png")
    if swimmer_file.exists():
        swimmer_raw = cv2.imread(str(swimmer_file))
        swimmer_scaled = cv2.resize(swimmer_raw, (70, 84), interpolation=cv2.INTER_LANCZOS4)
        
        # Elliptical soft mask
        s_mask = np.zeros((84, 70), dtype=np.uint8)
        cv2.ellipse(s_mask, (35, 42), (28, 36), 0, 0, 360, 255, -1)
        s_mask = cv2.GaussianBlur(s_mask, (7, 7), 0)
        
        # Center right where the swimmer belongs at the splash
        inpainted = cv2.seamlessClone(swimmer_scaled, inpainted, s_mask, (215, 245), cv2.NORMAL_CLONE)

    cv2.imwrite(str(dst_path), inpainted, [cv2.IMWRITE_JPEG_QUALITY, 95])
    print(f"✓ Cleaned Sample 01 saved to {dst_path}")
    return inpainted


def create_poisson_swimmer_scene(
    bg_img_path: Path,
    swimmer_specs: list,
    dst_path: Path,
    crop_center: tuple = None,
    crop_size: int = 512
):
    """
    Blends genuine, board-free aerial swimmers into rip current backgrounds.
    swimmer_specs: list of (swimmer_file, target_xy, scale_wh, angle_deg)
    """
    bg = cv2.imread(str(bg_img_path))
    if bg is None:
        raise FileNotFoundError(f"Cannot read {bg_img_path}")

    h, w = bg.shape[:2]

    # Crop focused around the rip current
    if crop_center is None:
        cx, cy = w // 2, h // 2
    else:
        cx, cy = crop_center

    x1 = max(0, min(w - crop_size, cx - crop_size // 2))
    y1 = max(0, min(h - crop_size, cy - crop_size // 2))
    canvas = bg[y1:y1 + crop_size, x1:x1 + crop_size].copy()

    for spec in swimmer_specs:
        sw_path, (target_x, target_y), (sw_w, sw_h), angle = spec
        sw_img = cv2.imread(str(sw_path))
        if sw_img is None:
            continue

        # Resize swimmer
        sw_resized = cv2.resize(sw_img, (sw_w, sw_h), interpolation=cv2.INTER_LANCZOS4)

        # Rotate if angle
        if angle != 0:
            M = cv2.getRotationMatrix2D((sw_w // 2, sw_h // 2), angle, 1.0)
            sw_resized = cv2.warpAffine(
                sw_resized, M, (sw_w, sw_h),
                borderMode=cv2.BORDER_REFLECT
            )

        # Create smooth elliptical mask
        mask = np.zeros((sw_h, sw_w), dtype=np.uint8)
        cv2.ellipse(mask, (sw_w // 2, sw_h // 2), (int(sw_w * 0.42), int(sw_h * 0.42)), 0, 0, 360, 255, -1)
        mask = cv2.GaussianBlur(mask, (7, 7), 0)

        # Bounds check
        target_x = max(sw_w // 2 + 5, min(crop_size - sw_w // 2 - 5, target_x))
        target_y = max(sw_h // 2 + 5, min(crop_size - sw_h // 2 - 5, target_y))

        try:
            canvas = cv2.seamlessClone(sw_resized, canvas, mask, (target_x, target_y), cv2.NORMAL_CLONE)
        except Exception as e:
            print(f"Seamless clone fallback: {e}")

    cv2.imwrite(str(dst_path), canvas, [cv2.IMWRITE_JPEG_QUALITY, 95])
    print(f"✓ Scene saved to {dst_path}")
    return canvas


def build_trial_collage(img_paths: list, out_path: Path):
    """Builds a formatted comparison collage of all 5 images."""
    card_w, card_h = 480, 480
    header_h = 42

    canvas_w = card_w * 3 + 40
    canvas_h = (card_h + header_h) * 2 + 30
    canvas = Image.new("RGB", (canvas_w, canvas_h), (20, 24, 30))
    draw = ImageDraw.Draw(canvas)

    coords = [
        (10, 10),
        (card_w + 20, 10),
        (card_w * 2 + 30, 10),
        (int(canvas_w / 2 - card_w - 5), card_h + header_h + 20),
        (int(canvas_w / 2 + 5), card_h + header_h + 20)
    ]

    labels = [
        "Sample 1: Swimmers Treading Water in Rip Corridor (Cleaned)",
        "Sample 2: Two Swimmers in Active Rip Neck & Surf Zone",
        "Sample 3: Swimmer Floating in Rip Trench Water Flow",
        "Sample 4: Swimmer Stroking Through Coastal Rip Plume",
        "Sample 5: Swimmer in Expanding Offshore Rip Head"
    ]

    for idx, (p, (x, y), label) in enumerate(zip(img_paths, coords, labels)):
        img = Image.open(p).convert("RGB").resize((card_w, card_h))
        draw.rectangle([(x, y), (x + card_w, y + header_h - 4)], fill=(34, 40, 52))
        draw.text((x + 12, y + 12), f"[{idx+1}] {label}", fill=(245, 245, 245))
        canvas.paste(img, (x, y + header_h))

    canvas.save(out_path, quality=95)
    print(f"✓ Collage saved to {out_path}")


def main():
    out_dir = Path("data/trial")
    out_dir.mkdir(parents=True, exist_ok=True)
    img_dir = out_dir / "images"
    img_dir.mkdir(parents=True, exist_ok=True)

    crops_dir = Path("data/afo_samples/a_1044_crops")
    sw_treading = crops_dir / "46_human_135x162.png"
    sw_floating = crops_dir / "30_human_151x134.png"
    sw_two = crops_dir / "21_human_115x97.png"
    sw_freestyle = crops_dir / "42_human_116x149.png"

    # 1. Sample 01: Cleaned version of Image 1 (blue block removed!)
    s1_path = img_dir / "trial_sample_01.jpg"
    clean_sample_01(img_dir / "trial_sample_01.jpg", s1_path)

    # 2. Sample 02: Two swimmers in rip neck & surf zone on RipVIS-004
    s2_path = img_dir / "trial_sample_02.jpg"
    create_poisson_swimmer_scene(
        Path("data/real_ripvis/train/images/RipVIS-004_00162.jpg"),
        [
            (sw_two, (240, 260), (95, 80), 20),
            (sw_treading, (310, 310), (70, 84), -15)
        ],
        s2_path,
        crop_center=(620, 380),
        crop_size=512
    )

    # 3. Sample 03: Swimmer floating in rip trench on RipVIS-017
    s3_path = img_dir / "trial_sample_03.jpg"
    create_poisson_swimmer_scene(
        Path("data/real_ripvis/train/images/RipVIS-017_00238.jpg"),
        [
            (sw_floating, (230, 240), (85, 75), 45),
            (sw_freestyle, (300, 320), (75, 95), -30)
        ],
        s3_path,
        crop_center=(560, 420),
        crop_size=512
    )

    # 4. Sample 04: Swimmer stroking through coastal rip current corridor
    s4_path = img_dir / "trial_sample_04.jpg"
    create_poisson_swimmer_scene(
        Path("data/real_ripvis/train/images/RipVIS-028_00282.jpg"),
        [
            (sw_freestyle, (250, 250), (80, 100), 10),
            (sw_treading, (330, 210), (70, 84), 0)
        ],
        s4_path,
        crop_center=(600, 400),
        crop_size=512
    )

    # 5. Sample 05: Swimmer in offshore rip head expanding seaward
    s5_path = img_dir / "trial_sample_05.jpg"
    create_poisson_swimmer_scene(
        Path("data/real_ripvis/train/images/RipVIS-004_00162.jpg"),
        [
            (sw_floating, (260, 220), (90, 80), -25)
        ],
        s5_path,
        crop_center=(640, 250),
        crop_size=512
    )

    # Build collage
    sample_paths = [s1_path, s2_path, s3_path, s4_path, s5_path]
    build_trial_collage(sample_paths, out_dir / "trial_5_samples.jpg")
    print("✓ All 5 clean trial samples successfully generated!")


if __name__ == "__main__":
    main()
