"""
generate_rip_with_afo_people.py
──────────────────────────────────
Combined pipeline:
1. Downloads & extracts real aerial human/floating object cutouts from Voxel51/AFO-Aerial_Floating_Objects on HF.
2. Imprints them onto synthetic rip current beach images with perspective scaling.
3. Generates COCO instance annotations for both rip currents and AFO people.

Usage:
    python scripts/generate_rip_with_afo_people.py --n_cutouts 40 --n_images 100
"""

import argparse
import json
import sys
from pathlib import Path
from typing import List, Tuple, Dict, Optional
import numpy as np
from PIL import Image, ImageFilter
from tqdm import tqdm
from datasets import load_dataset

sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.person_utils import place_person_cutout, calculate_perspective_scale
from utils.coco_utils import COCODatasetBuilder
from utils.mask_utils import load_mask, save_mask


def download_afo_cutouts(output_dir: Path, max_cutouts: int = 40) -> List[Path]:
    """Download and prepare real aerial floating object cutouts from AFO dataset."""
    output_dir.mkdir(parents=True, exist_ok=True)
    existing_cutouts = list(output_dir.glob("*.png"))
    if len(existing_cutouts) >= max_cutouts:
        print(f"Using {len(existing_cutouts)} existing AFO cutouts in '{output_dir}'.")
        return existing_cutouts[:max_cutouts]

    print(f"Downloading & extracting {max_cutouts} real AFO aerial cutouts from HuggingFace...")
    ds = load_dataset("Voxel51/AFO-Aerial_Floating_Objects", split="test", streaming=True)

    extracted_paths = []
    saved_count = 0
    allowed_labels = {"human", "swimmer", "person"}

    pbar = tqdm(total=max_cutouts, desc="Extracting AFO Cutouts")
    for sample in ds:
        if saved_count >= max_cutouts:
            break
        gt_raw = sample.get("ground_truth")
        if not gt_raw:
            continue
        try:
            gt_data = json.loads(gt_raw)
            dets = gt_data.get("detections", [])
        except Exception as e:
            print(f"JSON parse error: {e}")
            continue

        if not dets:
            continue

        img_pil = sample["image"].convert("RGB")
        w, h = img_pil.size

        for det in dets:
            if saved_count >= max_cutouts:
                break
            lbl = str(det.get("label", "")).lower()
            if lbl not in allowed_labels:
                continue

            bbox = det.get("bounding_box")
            if not bbox or len(bbox) < 4:
                continue

            rx, ry, rw, rh = bbox
            px_x1 = int(rx * w)
            px_y1 = int(ry * h)
            px_w  = int(rw * w)
            px_h  = int(rh * h)

            pad = max(25, int(max(px_w, px_h) * 1.2))
            x1 = max(0, px_x1 - pad)
            y1 = max(0, px_y1 - pad)
            x2 = min(w, px_x1 + px_w + pad)
            y2 = min(h, px_y1 + px_h + pad)

            crop_w = x2 - x1
            crop_h = y2 - y1
            if crop_w < 10 or crop_h < 10:
                continue

            crop = img_pil.crop((x1, y1, x2, y2)).convert("RGBA")

            # Soft water border feathering
            alpha_np = np.ones((crop_h, crop_w), dtype=np.float32) * 255.0
            border_px = max(4, int(min(crop_w, crop_h) * 0.25))

            for b in range(border_px):
                factor = float(b + 1) / float(border_px)
                alpha_np[b, :] *= factor
                alpha_np[crop_h - 1 - b, :] *= factor
                alpha_np[:, b] *= factor
                alpha_np[:, crop_w - 1 - b] *= factor

            alpha_pil = Image.fromarray(np.clip(alpha_np, 0, 255).astype(np.uint8)).filter(ImageFilter.GaussianBlur(radius=3))
            crop.putalpha(alpha_pil)

            out_file = output_dir / f"afo_{lbl}_{saved_count:03d}.png"
            crop.save(out_file)
            extracted_paths.append(out_file)
            saved_count += 1
            pbar.update(1)

    pbar.close()
    if not extracted_paths:
        print("Falling back to local/procedural person cutouts in 'data/person_cutouts'...")
        from utils.person_utils import load_person_cutouts
        load_person_cutouts("data/person_cutouts")
        extracted_paths = list(Path("data/person_cutouts").glob("*.png"))

    print(f"Extracted/loaded {len(extracted_paths)} cutouts to '{output_dir}'.")
    return extracted_paths


def main():
    parser = argparse.ArgumentParser(description="Combine Rip Current synthetic images with AFO Aerial Floating Objects.")
    parser.add_argument("--input_images", type=str, default="data/synthetic/imprinted/images")
    parser.add_argument("--input_masks", type=str, default="data/synthetic/imprinted/masks")
    parser.add_argument("--output_dir", type=str, default="data/synthetic_afo_swimmers")
    parser.add_argument("--afo_cutouts_dir", type=str, default="data/afo_cutouts")
    parser.add_argument("--n_cutouts", type=int, default=50)
    parser.add_argument("--max_images", type=int, default=250)
    parser.add_argument("--min_swimmers", type=int, default=1)
    parser.add_argument("--max_swimmers", type=int, default=5)
    parser.add_argument("--no_harmonize", action="store_true", help="Disable LAB color and relighting harmonization.")
    parser.add_argument("--no_deep_matting", action="store_true", help="Disable distance-guided deep alpha matting.")
    parser.add_argument("--save_vis", action="store_true", default=True, help="Save side-by-side preview visualizations.")
    args = parser.parse_args()

    input_img_dir = Path(args.input_images)
    input_mask_dir = Path(args.input_masks)
    output_dir = Path(args.output_dir)
    afo_dir = Path(args.afo_cutouts_dir)

    enable_harmonize = not args.no_harmonize
    enable_matting = not args.no_deep_matting

    out_images_dir = output_dir / "images"
    out_masks_dir = output_dir / "person_masks"
    out_vis_dir = output_dir / "visualizations"
    out_images_dir.mkdir(parents=True, exist_ok=True)
    out_masks_dir.mkdir(parents=True, exist_ok=True)
    if args.save_vis:
        out_vis_dir.mkdir(parents=True, exist_ok=True)

    # 1. Check existing or download AFO Cutouts
    cutout_paths = sorted(list(afo_dir.glob("*.png")))
    if not cutout_paths:
        cutout_paths = sorted(list(Path("data/afo_cutouts_a1044").glob("*.png")))
    if not cutout_paths:
        cutout_paths = sorted(list(Path("data/person_cutouts").glob("*.png")))
    if not cutout_paths:
        cutout_paths = download_afo_cutouts(afo_dir, max_cutouts=args.n_cutouts)

    if not cutout_paths:
        print("Error: No cutouts available.")
        return

    cutout_arrays = [np.array(Image.open(p).convert("RGBA")) for p in cutout_paths]
    print(f"Loaded {len(cutout_arrays)} real AFO swimmer cutouts into memory from '{cutout_paths[0].parent}'.")

    # 2. Setup COCO Builder
    categories = [
        {"id": 1, "name": "rip_current", "supercategory": "water_hazard"},
        {"id": 2, "name": "person",      "supercategory": "human"}
    ]
    coco_builder = COCODatasetBuilder(
        dataset_name="RipVIS-Synthetic-AFO-Swimmers",
        categories=categories
    )

    # 3. Find input images
    img_files = sorted(list(input_img_dir.glob("*.png")) + list(input_img_dir.glob("*.jpg")))
    mask_files = sorted(list(input_mask_dir.glob("*.png")))

    if args.max_images:
        img_files = img_files[:args.max_images]

    rng = np.random.default_rng(42)
    print(f"Processing {len(img_files)} synthetic rip current images...")
    print(f"  Color Harmonization: {enable_harmonize} | Deep Alpha Matting: {enable_matting}")

    import cv2

    for idx, img_path in enumerate(tqdm(img_files, desc="Imprinting AFO Swimmers")):
        img_pil = Image.open(img_path).convert("RGB")
        img_np = np.array(img_pil)
        h, w = img_np.shape[:2]

        out_name = f"rip_afo_{idx:04d}_{img_path.name}"
        img_id = coco_builder.add_image(image_path=img_path, file_name=out_name)

        # Rip mask registration
        rip_mask = None
        if idx < len(mask_files) and mask_files[idx].exists():
            rip_mask = load_mask(mask_files[idx])
            coco_builder.add_annotation_from_mask(image_id=img_id, mask=rip_mask, category_id=1)

        # Imprint 1 to max_swimmers AFO swimmers per image
        n_people = rng.integers(args.min_swimmers, args.max_swimmers + 1)
        comp_np = img_np.copy()
        combined_person_mask = np.zeros((h, w), dtype=np.uint8)
        bboxes = []

        # Find rip current coordinates if present for biased placement
        rip_pts = np.argwhere(rip_mask > 0) if (rip_mask is not None and np.sum(rip_mask) > 0) else None

        for _ in range(n_people):
            cutout_rgba = cutout_arrays[rng.integers(0, len(cutout_arrays))]

            # 40% probability to place swimmer directly in/near rip current channel
            if rip_pts is not None and len(rip_pts) > 0 and rng.uniform() < 0.40:
                pt_idx = rng.integers(0, len(rip_pts))
                cy, cx = rip_pts[pt_idx]
                cx += int(rng.normal(0, 15))
                cy += int(rng.normal(0, 15))
            else:
                cx = rng.integers(int(w * 0.15), int(w * 0.85))
                cy = rng.integers(int(h * 0.28), int(h * 0.88))

            target_h = int(h * calculate_perspective_scale(cy, h, min_scale=0.07, max_scale=0.18))
            rotation = float(rng.uniform(-180.0, 180.0))

            comp_np, p_mask, bbox = place_person_cutout(
                background_img=comp_np,
                cutout_rgba=cutout_rgba,
                center_pos=(cx, cy),
                target_height_px=target_h,
                rotation_deg=rotation,
                enable_harmonization=enable_harmonize,
                enable_deep_matting=enable_matting,
                harmonization_strength=0.70
            )

            if np.sum(p_mask) > 0:
                combined_person_mask = np.maximum(combined_person_mask, p_mask)
                coco_builder.add_annotation_from_mask(image_id=img_id, mask=p_mask, category_id=2, min_area=40)
                bboxes.append(bbox)

        # Save output image and person mask
        out_img_file = out_images_dir / out_name
        out_mask_file = out_masks_dir / f"mask_{idx:04d}.png"
        Image.fromarray(comp_np).save(out_img_file)
        save_mask(combined_person_mask, out_mask_file)

        # Save sample visual previews
        if args.save_vis and idx < 20:
            vis_img = comp_np.copy()
            if rip_mask is not None:
                contours, _ = cv2.findContours(rip_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                cv2.drawContours(vis_img, contours, -1, (255, 60, 60), 2)
            for (bx, by, bw, bh) in bboxes:
                cv2.rectangle(vis_img, (bx, by), (bx + bw, by + bh), (40, 255, 80), 2)
                cv2.putText(vis_img, "swimmer", (bx, max(12, by - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (40, 255, 80), 1)

            vis_side = np.hstack([img_np, vis_img])
            Image.fromarray(vis_side).save(out_vis_dir / f"preview_{idx:03d}.jpg", quality=95)

    # Save COCO JSON
    json_path = output_dir / "synthetic_afo_coco.json"
    coco_builder.save(json_path)
    print(f"\nCompleted! Saved generated images & annotations to '{output_dir}'.")
    print(f"COCO JSON: {json_path}")


if __name__ == "__main__":
    main()
