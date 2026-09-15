"""
imprint_people.py
──────────────────
Method 3: Hybrid Person Insertion & Refinement Pipeline

Imprints person cutouts (swimmers, surfers, waders) onto synthetic beach/rip current images
using perspective-aware scaling and composites, followed by an optional light diffusion
inpainting refinement pass to generate natural water splashes, wake, and lighting integration.

Saves multi-class COCO annotations:
  - category_id 1: rip_current
  - category_id 2: person

Usage:
    # Quick test (uses procedural cutouts and CPU/synthetic demo)
    python scripts/imprint_people.py --test

    # Full run
    python scripts/imprint_people.py \
        --input_dir data/synthetic \
        --mask_dir data/masks \
        --output_dir data/synthetic_people \
        --cutouts_dir data/person_cutouts \
        --use_diffusion
"""

import argparse
import sys
import os
from pathlib import Path
from typing import List, Tuple, Dict, Optional, Union

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import cv2
from PIL import Image
from tqdm import tqdm
import yaml

from utils.mask_utils import (
    load_mask, save_mask, load_image_and_mask, get_mask_area
)
from utils.person_utils import (
    load_person_cutouts, place_person_cutout,
    calculate_perspective_scale, create_refinement_mask
)
from utils.coco_utils import COCODatasetBuilder


# ──────────────────────────────────────────────────────────────────────────────
# Pipeline Implementation
# ──────────────────────────────────────────────────────────────────────────────

def run_hybrid_person_imprinting(
    image_paths: List[Path],
    rip_mask_paths: Optional[List[Path]] = None,
    output_dir: Union[str, Path] = "data/synthetic_people",
    cutouts_dir: Union[str, Path] = "data/person_cutouts",
    max_people: int = 3,
    min_people: int = 1,
    use_diffusion: bool = False,
    refinement_strength: float = 0.35,
    device: str = "auto",
    coco_output_path: Optional[Union[str, Path]] = None,
    seed: int = 42
) -> Dict[str, int]:
    """
    Imprints people into a collection of synthetic beach images and optionally refines them
    with Stable Diffusion inpainting. Registers multi-class COCO annotations.

    Returns:
        Summary statistics dict.
    """
    rng = np.random.default_rng(seed)
    output_dir = Path(output_dir)
    images_out_dir = output_dir / "images"
    masks_out_dir = output_dir / "person_masks"
    images_out_dir.mkdir(parents=True, exist_ok=True)
    masks_out_dir.mkdir(parents=True, exist_ok=True)

    cutouts = load_person_cutouts(cutouts_dir)
    cutout_filenames = [p.name for p in Path(cutouts_dir).glob("*.png")]
    print(f"Loaded {len(cutouts)} person cutout templates from '{cutouts_dir}'. (Sample files: {cutout_filenames[:5]})")

    # Initialize COCO Builder for 2 categories
    categories = [
        {"id": 1, "name": "rip_current", "supercategory": "water_hazard"},
        {"id": 2, "name": "person",      "supercategory": "human"}
    ]
    coco_builder = COCODatasetBuilder(
        dataset_name="RipVIS-Synthetic-People",
        categories=categories
    )

    # Initialize SD Pipeline if refinement requested
    pipe = None
    if use_diffusion:
        try:
            import torch
            from diffusers import StableDiffusionInpaintPipeline
            model_id = "stabilityai/stable-diffusion-2-inpainting"
            print(f"Loading SD Inpainting model '{model_id}' for refinement pass...")
            
            dtype = torch.float16 if (torch.cuda.is_available() and device != "cpu") else torch.float32
            dev = "cuda" if (torch.cuda.is_available() and device != "cpu") else "cpu"
            pipe = StableDiffusionInpaintPipeline.from_pretrained(
                model_id, torch_dtype=dtype
            ).to(dev)
            if dev == "cuda":
                pipe.enable_attention_slicing()
            print(f"SD Inpainting ready on {dev}.")
        except Exception as e:
            print(f"Warning: Could not load SD Inpainting pipeline ({e}). Proceeding without diffusion refinement.")
            pipe = None

    stats = {"processed_images": 0, "total_people_inserted": 0, "rip_masks_registered": 0}

    for idx, img_path in enumerate(tqdm(image_paths, desc="Inserting People")):
        img_pil = Image.open(img_path).convert("RGB")
        img_np = np.array(img_pil)
        h, w = img_np.shape[:2]

        # Register image in COCO
        out_img_name = f"person_synth_{idx:04d}_{img_path.name}"
        img_id = coco_builder.add_image(
            image_path=img_path,
            file_name=out_img_name
        )

        # 1. Register existing rip current mask if available
        rip_mask = None
        if rip_mask_paths and idx < len(rip_mask_paths) and rip_mask_paths[idx].exists():
            rip_mask = load_mask(rip_mask_paths[idx])
            coco_builder.add_annotation_from_mask(
                image_id=img_id,
                mask=rip_mask,
                category_id=1,
                min_area=300
            )
            stats["rip_masks_registered"] += 1

        # 2. Select number of people to insert
        n_people = rng.integers(min_people, max_people + 1)
        composite_np = img_np.copy()
        combined_person_mask = np.zeros((h, w), dtype=np.uint8)

        for p_idx in range(n_people):
            cutout_rgba = cutouts[rng.integers(0, len(cutouts))]

            # Pick target coordinates (bias towards water zone in top 70% of frame)
            cx = rng.integers(int(w * 0.15), int(w * 0.85))
            cy = rng.integers(int(h * 0.25), int(h * 0.85))

            # Compute scale factor from perspective
            target_h = int(h * calculate_perspective_scale(cy, h, min_scale=0.07, max_scale=0.22))
            rotation = float(rng.uniform(-15.0, 15.0))

            composite_np, p_mask, bbox = place_person_cutout(
                background_img=composite_np,
                cutout_rgba=cutout_rgba,
                center_pos=(cx, cy),
                target_height_px=target_h,
                rotation_deg=rotation
            )

            if np.sum(p_mask) > 0:
                combined_person_mask = np.maximum(combined_person_mask, p_mask)

                # Register person in COCO
                coco_builder.add_annotation_from_mask(
                    image_id=img_id,
                    mask=p_mask,
                    category_id=2,
                    min_area=100
                )
                stats["total_people_inserted"] += 1

        # 3. Method 3 Diffusion Refinement Pass
        if pipe is not None and np.sum(combined_person_mask) > 0:
            refine_mask_np = create_refinement_mask(combined_person_mask, dilate_radius=20)
            refine_pil_img = Image.fromarray(composite_np)
            refine_pil_mask = Image.fromarray(refine_mask_np)

            prompt = (
                "swimmer in ocean water, body in waves, natural white water splash, "
                "realistic lighting, photorealistic beach photography"
            )
            negative_prompt = "cartoon, low quality, unnatural borders, floating limbs"

            try:
                import torch
                generator = torch.Generator(device=pipe.device).manual_seed(seed + idx)
                refined = pipe(
                    prompt=prompt,
                    negative_prompt=negative_prompt,
                    image=refine_pil_img,
                    mask_image=refine_pil_mask,
                    strength=refinement_strength,
                    num_inference_steps=20,
                    guidance_scale=7.0,
                    generator=generator
                ).images[0]

                composite_np = np.array(refined.resize((w, h)))
            except Exception as e:
                print(f"Refinement step failed for {img_path.name}: {e}")

        # Save outputs
        out_img_path = images_out_dir / out_img_name
        out_mask_path = masks_out_dir / f"person_mask_{idx:04d}.png"

        Image.fromarray(composite_np).save(out_img_path)
        save_mask(combined_person_mask, out_mask_path)

        stats["processed_images"] += 1

    # Save COCO Annotations
    if coco_output_path is None:
        coco_output_path = output_dir / "synthetic_people_coco.json"
    coco_builder.save(coco_output_path)
    print(f"\nSaved multi-class COCO annotations to: {coco_output_path}")

    return stats


# ──────────────────────────────────────────────────────────────────────────────
# Quick Test / CLI Entrypoint
# ──────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Imprint people onto synthetic beach images (Method 3).")
    parser.add_argument("--input_dir", type=str, default="data/synthetic", help="Directory of synthetic frames.")
    parser.add_argument("--mask_dir", type=str, default="data/masks", help="Directory of rip masks.")
    parser.add_argument("--output_dir", type=str, default="data/synthetic_people", help="Output directory.")
    parser.add_argument("--cutouts_dir", type=str, default="data/person_cutouts", help="Directory of person cutouts.")
    parser.add_argument("--max_people", type=int, default=3, help="Max people per frame.")
    parser.add_argument("--use_diffusion", action="store_true", help="Run SD inpainting refinement pass.")
    parser.add_argument("--test", action="store_true", help="Run quick demo test.")
    args = parser.parse_args()

    if args.test:
        print("=== Running Quick Person Imprinting Test ===")
        # Create temp test directory
        test_dir = Path("data/synthetic_people_test")
        test_dir.mkdir(parents=True, exist_ok=True)

        # Generate a synthetic demo background image
        from scripts.download_samples import create_demo_beach_images
        demo_dir = test_dir / "demo_base"
        create_demo_beach_images(demo_dir, n=2)

        demo_images = list((demo_dir / "images").glob("*.png"))
        demo_masks = list((demo_dir / "masks").glob("*.png"))

        stats = run_hybrid_person_imprinting(
            image_paths=demo_images,
            rip_mask_paths=demo_masks,
            output_dir=test_dir,
            cutouts_dir="data/person_cutouts",
            max_people=3,
            use_diffusion=args.use_diffusion
        )
        print("\nTest Run Complete Statistics:", stats)
        return

    # Standard run
    input_dir = Path(args.input_dir)
    mask_dir = Path(args.mask_dir) if args.mask_dir else None

    image_paths = sorted(list(input_dir.glob("*.png")) + list(input_dir.glob("*.jpg")))
    mask_paths = sorted(list(mask_dir.glob("*.png"))) if mask_dir and mask_dir.exists() else []

    if not image_paths:
        print(f"No input images found in '{input_dir}'. Searching in subdirectories...")
        image_paths = sorted(list(input_dir.rglob("*.png")) + list(input_dir.rglob("*.jpg")))

    if not image_paths:
        print(f"Error: No images found. Run 'python scripts/imprint_people.py --test' to test with demo data.")
        return

    stats = run_hybrid_person_imprinting(
        image_paths=image_paths,
        rip_mask_paths=mask_paths,
        output_dir=args.output_dir,
        cutouts_dir=args.cutouts_dir,
        max_people=args.max_people,
        use_diffusion=args.use_diffusion
    )
    print("\nHybrid Person Imprinting Completed:", stats)


if __name__ == "__main__":
    main()
