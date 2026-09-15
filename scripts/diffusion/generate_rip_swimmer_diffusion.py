"""
generate_rip_swimmer_diffusion.py
─────────────────────────────────
Diffusion-based generation and inpainting pipeline for swimmers near rip currents.

Modes:
1. Text-to-Image (T2I): Generates full photorealistic aerial scenes of rip currents + swimmers using fine-tuned LoRA.
2. Rip-Aware Inpainting: Inpaints realistic swimmers into real or synthetic rip current frames near the foam corridor.
3. Automatic Dataset & COCO/YOLO Annotation Generation.

Usage:
    # Mode 1: Text-to-Image Generation
    python scripts/generate_rip_swimmer_diffusion.py \
        --mode t2i \
        --lora_path models/diffusion_rip_swimmer_lora/final_lora \
        --output_dir data/synthetic_diffusion/ \
        --n_images 16 \
        --steps 35 \
        --guidance 7.0

    # Mode 2: Rip-Aware Inpainting on existing frames
    python scripts/generate_rip_swimmer_diffusion.py \
        --mode inpaint \
        --input_dir data/samples/ \
        --output_dir data/synthetic_diffusion_inpainted/ \
        --n_images 16
"""

import argparse
import json
import math
import os
import random
import sys
from pathlib import Path
from typing import List, Tuple, Optional, Dict, Any

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFilter
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))
from utils.coco_utils import COCODatasetBuilder


# High quality photographic prompts for rip currents + swimmers
DEFAULT_PROMPTS = [
    "aerial drone photograph of a swimmer caught in the turbulent foam channel of an outgoing rip current, turquoise ocean water, white foam plumes, natural sunlight caustics, Hasselblad marine photography, 4k photorealistic, sharp focus",
    "top-down aerial view of two swimmers treading water in coastal surf zone near a dangerous dark rip current corridor, breaking wave crests, realistic human body anatomy, natural water displacement, high quality photography",
    "overhead drone view of an ocean beach surf zone with a prominent foamy rip current neck pulling seaward, swimmer nearby with subtle wake ripples, photorealistic ocean optics, natural water lighting",
    "drone perspective of a person swimming freestyle in deep ocean water adjacent to a foamy rip current plume, clear coastal water caustics, realistic subsurface refraction, crisp marine details",
    "aerial view of breaking beach waves with a foamy rip current head extending outward into deep water, swimmer floating near the current boundary, natural marine lighting, high resolution",
    "top-down drone capture of a turbulent outgoing rip current neck with swimmers in turquoise coastal water, natural sun reflections, crisp focus, photorealistic",
    "aerial drone view of a swimmer in open ocean water navigating near a foamy rip current corridor, authentic water foam dynamics, photorealistic marine optics",
]

INPAINT_SWIMMER_PROMPTS = [
    "top-down aerial drone photograph of a swimmer floating in turquoise ocean water, realistic human body anatomy, subtle foam trails, photorealistic marine optics, natural water caustics and subsurface light refraction",
    "aerial drone view of a person swimming freestyle in coastal sea water near surf foam, natural water displacement, photorealistic marine lighting, crisp ocean texture",
    "top-down drone view of a swimmer treading water in coastal ocean, realistic sun reflection, natural ripples and micro foam wake, high quality photography",
    "overhead aerial photograph of a swimmer in clear turquoise coastal water with gentle wave distortion, natural skin tone, photorealistic marine optics",
]

NEGATIVE_PROMPT = (
    "cartoon, 3d render, illustration, drawing, painting, bad anatomy, deformed swimmer, "
    "floating cutout, sticker effect, plastic skin, blurry, low quality, unnatural water, fake ocean, bad perspective, text, watermark, "
    "oversaturated, pixelated, duplicated body, extra limbs, missing limbs, artificial border, CGI"
)


def get_pipeline_device():
    return "cuda" if torch.cuda.is_available() else "cpu"


def configure_scheduler(pipe, scheduler_name: str = "dpmsolver"):
    """Configure fast and high-fidelity sampling scheduler."""
    from diffusers import DPMSolverMultistepScheduler, EulerAncestralDiscreteScheduler, DDIMScheduler
    if scheduler_name == "dpmsolver":
        pipe.scheduler = DPMSolverMultistepScheduler.from_config(
            pipe.scheduler.config,
            use_karras_sigmas=True,
            algorithm_type="sde-dpmsolver++"
        )
    elif scheduler_name == "euler_a":
        pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(pipe.scheduler.config)
    elif scheduler_name == "ddim":
        pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
    return pipe


def create_inpaint_mask(img_w: int, img_h: int, rip_mask_np: Optional[np.ndarray] = None) -> Tuple[Image.Image, Tuple[int, int, int, int]]:
    """Create a realistic bounding mask for inpainting a swimmer in the ocean water."""
    mask = Image.new("L", (img_w, img_h), 0)
    draw = ImageDraw.Draw(mask)

    swimmer_w = random.randint(int(img_w * 0.04), int(img_w * 0.08))
    swimmer_h = random.randint(int(img_h * 0.04), int(img_h * 0.08))

    if rip_mask_np is not None and np.any(rip_mask_np > 0):
        ys, xs = np.where(rip_mask_np > 0)
        center_x = int(np.mean(xs)) + random.randint(-int(img_w * 0.15), int(img_w * 0.15))
        center_y = int(np.mean(ys)) + random.randint(-int(img_h * 0.15), int(img_h * 0.15))
    else:
        center_x = random.randint(int(img_w * 0.25), int(img_w * 0.75))
        center_y = random.randint(int(img_h * 0.25), int(img_h * 0.75))

    x1 = max(0, center_x - swimmer_w // 2)
    y1 = max(0, center_y - swimmer_h // 2)
    x2 = min(img_w, x1 + swimmer_w)
    y2 = min(img_h, y1 + swimmer_h)

    draw.ellipse([x1, y1, x2, y2], fill=255)
    mask = mask.filter(ImageFilter.GaussianBlur(radius=4))

    return mask, (x1, y1, x2 - x1, y2 - y1)


def save_grid_preview(images: List[Image.Image], output_path: Path, n_cols: int = 4, thumb_size: int = 384):
    """Save a clean visual grid preview of generated images."""
    if not images:
        return
    n_images = len(images)
    n_cols = min(n_cols, n_images)
    n_rows = math.ceil(n_images / n_cols)
    pad = 12

    grid_w = n_cols * thumb_size + (n_cols + 1) * pad
    grid_h = n_rows * thumb_size + (n_rows + 1) * pad

    canvas = Image.new("RGB", (grid_w, grid_h), (18, 22, 28))
    for i, im in enumerate(images):
        col = i % n_cols
        row = i // n_cols
        x = pad + col * (thumb_size + pad)
        y = pad + row * (thumb_size + pad)
        im_resized = im.resize((thumb_size, thumb_size), Image.Resampling.LANCZOS)
        canvas.paste(im_resized, (x, y))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path, "JPEG", quality=95)
    print(f"✓ Grid summary preview saved to: {output_path}")


def run_t2i_generation(args):
    """Generate full synthetic scenes with text-to-image pipeline."""
    from diffusers import StableDiffusionPipeline

    device = get_pipeline_device()
    dtype = torch.float16 if device == "cuda" else torch.float32

    print(f"Loading Base SD Model: {args.base_model} on {device}...")
    pipe = StableDiffusionPipeline.from_pretrained(
        args.base_model,
        torch_dtype=dtype,
        safety_checker=None
    )

    pipe = configure_scheduler(pipe, args.scheduler)

    if args.lora_path and Path(args.lora_path).exists():
        print(f"Applying fine-tuned LoRA adapter from: {args.lora_path}...")
        try:
            pipe.load_lora_weights(args.lora_path)
            print("✓ LoRA adapter loaded successfully!")
        except Exception as e:
            print(f"Warning loading LoRA: {e}, falling back to base model.")

    pipe.to(device)
    if device == "cuda":
        pipe.enable_attention_slicing()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    images_dir = output_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)

    coco_builder = COCODatasetBuilder(
        dataset_name="RipVIS Diffusion-Generated Rip Currents & Swimmers Dataset",
        categories=[{"id": 1, "name": "rip_current", "supercategory": "hazard"}, {"id": 2, "name": "swimmer", "supercategory": "human"}]
    )

    generated_images = []
    print(f"Generating {args.n_images} photorealistic scenes via Diffusion ({args.scheduler}, steps={args.steps}, cfg={args.guidance})...")
    for idx in tqdm(range(args.n_images), desc="Generating Images"):
        prompt = DEFAULT_PROMPTS[idx % len(DEFAULT_PROMPTS)]
        generator = torch.Generator(device=device).manual_seed(args.seed + idx)

        image = pipe(
            prompt=prompt,
            negative_prompt=NEGATIVE_PROMPT,
            num_inference_steps=args.steps,
            guidance_scale=args.guidance,
            generator=generator,
            height=args.img_size,
            width=args.img_size,
        ).images[0]

        img_filename = f"diff_t2i_{idx:04d}.jpg"
        img_save_path = images_dir / img_filename
        image.save(img_save_path, "JPEG", quality=95)
        generated_images.append(image)

        coco_builder.add_image(image_path=img_save_path)

    # Save COCO JSON
    coco_path = output_dir / "synthetic_diffusion_coco.json"
    coco_builder.save(coco_path)
    print(f"\n✓ T2I Generation complete! Saved {args.n_images} images to {output_dir}")

    # Save preview grid
    grid_path = output_dir / "generation_preview_grid.jpg"
    save_grid_preview(generated_images, grid_path, n_cols=4)


def run_inpaint_generation(args):
    """Inpaint realistic swimmers into existing rip current frames."""
    from diffusers import AutoPipelineForInpainting

    device = get_pipeline_device()
    dtype = torch.float16 if device == "cuda" else torch.float32

    print(f"Loading Inpainting Pipeline on {device}...")
    pipe = AutoPipelineForInpainting.from_pretrained(
        "runwayml/stable-diffusion-v1-5",
        torch_dtype=dtype,
        safety_checker=None
    )

    pipe = configure_scheduler(pipe, args.scheduler)

    if args.lora_path and Path(args.lora_path).exists():
        print(f"Loading LoRA adapter: {args.lora_path}...")
        try:
            pipe.load_lora_weights(args.lora_path)
            print("✓ LoRA adapter loaded into inpainting pipeline!")
        except Exception as e:
            print(f"LoRA loading note: {e}")

    pipe.to(device)
    if device == "cuda":
        pipe.enable_attention_slicing()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    images_dir = output_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)

    input_files = list(Path(args.input_dir).glob("*.jpg")) + list(Path(args.input_dir).glob("*.png"))
    if not input_files:
        input_files = list(Path(args.input_dir).rglob("*.jpg"))[:100]

    if not input_files:
        for fallback in [Path("data/real_ripvis/train/images"), Path("data/diffusion_train/images")]:
            if fallback.exists():
                input_files = list(fallback.glob("*.jpg"))[:100]
                if input_files:
                    print(f"Using {len(input_files)} real rip current frames from {fallback}.")
                    break

    if not input_files:
        print(f"No input images found in {args.input_dir}. Using synthetic blanks.")
        input_files = []

    coco_builder = COCODatasetBuilder(
        dataset_name="RipVIS Inpainted Swimmers & Rip Currents Dataset",
        categories=[{"id": 1, "name": "rip_current", "supercategory": "hazard"}, {"id": 2, "name": "swimmer", "supercategory": "human"}]
    )

    count = min(args.n_images, len(input_files)) if input_files else args.n_images
    print(f"Inpainting swimmers into {count} frames ({args.scheduler})...")

    generated_images = []
    for idx in tqdm(range(count), desc="Inpainting Swimmers"):
        if input_files:
            base_img = Image.open(input_files[idx % len(input_files)]).convert("RGB")
            base_img = base_img.resize((args.img_size, args.img_size), Image.Resampling.LANCZOS)
        else:
            base_img = Image.new("RGB", (args.img_size, args.img_size), (30, 80, 140))

        mask_img, (sx, sy, sw, sh) = create_inpaint_mask(args.img_size, args.img_size)
        prompt = random.choice(INPAINT_SWIMMER_PROMPTS)
        generator = torch.Generator(device=device).manual_seed(args.seed + idx)

        result_img = pipe(
            prompt=prompt,
            negative_prompt=NEGATIVE_PROMPT,
            image=base_img,
            mask_image=mask_img,
            num_inference_steps=args.steps,
            guidance_scale=args.guidance,
            generator=generator,
        ).images[0]

        img_filename = f"diff_inpaint_{idx:04d}.jpg"
        img_save_path = images_dir / img_filename
        result_img.save(img_save_path, "JPEG", quality=95)
        generated_images.append(result_img)

        # Register COCO annotation
        img_id = coco_builder.add_image(image_path=img_save_path)
        coco_builder.add_annotation_from_mask(
            image_id=img_id,
            mask=np.array(mask_img),
            category_id=2,  # swimmer
            min_area=10
        )

    coco_path = output_dir / "synthetic_diffusion_inpainted_coco.json"
    coco_builder.save(coco_path)
    print(f"\n✓ Inpainting complete! Saved results to {output_dir}")

    # Save preview grid
    grid_path = output_dir / "inpainting_preview_grid.jpg"
    save_grid_preview(generated_images, grid_path, n_cols=4)


def parse_args():
    p = argparse.ArgumentParser(description="Generate swimmers near rip currents using Diffusion.")
    p.add_argument("--mode", type=str, default="t2i", choices=["t2i", "inpaint"],
                   help="Generation mode: t2i (text-to-image) or inpaint (inpaint into existing frame).")
    p.add_argument("--base_model", type=str, default="runwayml/stable-diffusion-v1-5")
    p.add_argument("--lora_path", type=str, default="models/diffusion_rip_swimmer_lora/final_lora")
    p.add_argument("--scheduler", type=str, default="dpmsolver", choices=["dpmsolver", "euler_a", "ddim"],
                   help="Sampling scheduler.")
    p.add_argument("--input_dir", type=str, default="data/samples")
    p.add_argument("--output_dir", type=str, default="data/synthetic_diffusion")
    p.add_argument("--n_images", type=int, default=16)
    p.add_argument("--img_size", type=int, default=512)
    p.add_argument("--steps", type=int, default=35)
    p.add_argument("--guidance", type=float, default=7.0)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.mode == "t2i":
        run_t2i_generation(args)
    else:
        run_inpaint_generation(args)
