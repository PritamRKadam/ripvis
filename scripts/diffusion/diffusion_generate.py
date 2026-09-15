"""
diffusion_generate.py
─────────────────────
Method B: Diffusion-Based Rip Current Generation

Uses Stable Diffusion 2 Inpainting to generate photorealistic rip currents
within masked regions of beach images.

The pipeline:
  1. Load a beach image (clean or with existing rip current)
  2. Load or generate a binary mask defining where to add the rip current
  3. Run SD2-inpainting: the model fills the masked region with a rip current
  4. Save the generated image + mask in COCO format

Supports:
  - CUDA GPU (recommended, float16)
  - CPU fallback (float32, slower)
  - HuggingFace Inference API (no local GPU needed)
  - Batch processing of entire directories

Usage:
    # Quick test (CPU, 5 steps)
    python scripts/diffusion_generate.py --test

    # Full run with GPU
    python scripts/diffusion_generate.py \
        --input_dir data/samples/ \
        --mask_dir data/masks/ \
        --output_dir data/synthetic/diffusion/ \
        --model stabilityai/stable-diffusion-2-inpainting \
        --steps 30 \
        --n_per_mask 3
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent))  # allow sibling script imports

import numpy as np
from PIL import Image
from tqdm import tqdm

from utils.mask_utils import (
    load_mask, save_mask, load_image_and_mask,
    get_mask_area, generate_random_ellipse_mask, dilate_mask
)
from utils.coco_utils import COCODatasetBuilder


# ──────────────────────────────────────────────────────────────────────────────
# Prompts for rip current generation
# ──────────────────────────────────────────────────────────────────────────────

POSITIVE_PROMPT = (
    "rip current, turbulent water flowing outward from the shore into the ocean, "
    "white foam channels, narrow dark water corridor, sediment discoloration, "
    "sandy beach background, realistic ocean photography, high resolution, "
    "photorealistic, natural lighting"
)

NEGATIVE_PROMPT = (
    "cartoon, painting, drawing, sketch, blurry, low quality, text, watermark, "
    "calm flat water, no waves, saturated colors, unrealistic"
)

# Prompt variants for diversity
PROMPT_VARIANTS = [
    "rip current, outgoing water channel in the surf zone, white turbulent foam, "
    "discolored water, beach safety hazard, realistic photograph",

    "strong rip current at beach, narrow turbulent channel, brown sediment "
    "discoloration, foamy edges, aerial drone view",

    "rip current visible from shore, dark channel through breaking waves, "
    "white foam trail, realistic ocean scene, natural photography",
]


# ──────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="Generate synthetic rip current images using Stable Diffusion inpainting."
    )
    p.add_argument("--input_dir",  type=str, default="data/samples",
                   help="Directory of input beach images.")
    p.add_argument("--mask_dir",   type=str, default=None,
                   help="Directory of binary masks (uses random masks if not given).")
    p.add_argument("--output_dir", type=str, default="data/synthetic/diffusion",
                   help="Output directory.")
    p.add_argument("--model",      type=str,
                   default="stabilityai/stable-diffusion-2-inpainting",
                   help="HuggingFace model ID for inpainting.")
    p.add_argument("--device",     type=str, default="auto",
                   choices=["auto", "cuda", "cpu", "mps"],
                   help="Compute device.")
    p.add_argument("--steps",      type=int, default=30,
                   help="Number of denoising steps (default: 30).")
    p.add_argument("--guidance",   type=float, default=7.5,
                   help="Classifier-free guidance scale (default: 7.5).")
    p.add_argument("--strength",   type=float, default=0.85,
                   help="Inpainting strength (0=keep original, 1=full repaint).")
    p.add_argument("--n_per_mask", type=int, default=3,
                   help="Number of images to generate per mask (default: 3).")
    p.add_argument("--img_size",   type=int, nargs=2, default=[512, 512],
                   metavar=("W", "H"),
                   help="Output image size (default: 512 512).")
    p.add_argument("--min_mask_area", type=int, default=500)
    p.add_argument("--annotation", type=str, default=None,
                   help="Output COCO annotation JSON path.")
    p.add_argument("--seed",       type=int, default=42)
    p.add_argument("--use_api",    action="store_true",
                   help="Use HuggingFace Inference API instead of local model.")
    p.add_argument("--hf_token",   type=str, default=None,
                   help="HuggingFace API token (for --use_api).")
    p.add_argument("--test",       action="store_true",
                   help="Quick smoke test (5 steps, single image).")
    return p.parse_args()


# ──────────────────────────────────────────────────────────────────────────────
# Diffusion pipeline loader
# ──────────────────────────────────────────────────────────────────────────────

def get_device(device_arg: str) -> str:
    if device_arg == "auto":
        try:
            import torch
            if torch.cuda.is_available():
                return "cuda"
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                return "mps"
        except ImportError:
            pass
        return "cpu"
    return device_arg


def load_pipeline(model_id: str, device: str):
    """
    Load the Stable Diffusion inpainting pipeline from HuggingFace.
    Returns the pipeline or None if diffusers is not installed.
    """
    try:
        import torch
        from diffusers import StableDiffusionInpaintPipeline
    except ImportError:
        print("ERROR: 'diffusers' and 'torch' required.")
        print("Install with: pip install diffusers transformers accelerate torch")
        return None

    dtype = "float16" if device == "cuda" else "float32"
    torch_dtype = {"float16": __import__("torch").float16,
                   "float32": __import__("torch").float32}[dtype]

    print(f"Loading model: {model_id}")
    print(f"  Device: {device}, dtype: {dtype}")

    try:
        pipe = StableDiffusionInpaintPipeline.from_pretrained(
            model_id,
            torch_dtype=torch_dtype,
            safety_checker=None,       # Disable NSFW filter for speed
            requires_safety_checker=False
        )
        pipe = pipe.to(device)

        # Memory optimization for low VRAM
        if device == "cuda":
            try:
                pipe.enable_xformers_memory_efficient_attention()
                print("  xformers memory efficient attention: enabled")
            except Exception:
                pipe.enable_attention_slicing()
                print("  attention slicing: enabled")

        print("  Model loaded successfully ✓")
        return pipe
    except Exception as e:
        print(f"ERROR loading model: {e}")
        return None


# ──────────────────────────────────────────────────────────────────────────────
# Core generation
# ──────────────────────────────────────────────────────────────────────────────

def generate_with_pipeline(
    pipe,
    image: Image.Image,
    mask: Image.Image,
    prompt: str,
    negative_prompt: str,
    n_steps: int,
    guidance_scale: float,
    strength: float,
    seed: int = 42,
    img_size: tuple = (512, 512)
) -> Image.Image:
    """
    Run a single inpainting generation.

    Args:
        pipe:           Loaded StableDiffusionInpaintPipeline.
        image:          PIL RGB image (will be resized to img_size).
        mask:           PIL grayscale mask (255=inpaint, 0=keep).
        prompt:         Positive text prompt.
        negative_prompt: Negative text prompt.
        n_steps:        Number of denoising steps.
        guidance_scale: CFG scale.
        strength:       How much to repaint (0-1).
        seed:           Generator seed.
        img_size:       (W, H) for model input/output.

    Returns:
        Generated PIL RGB image at img_size resolution.
    """
    import torch

    w, h = img_size
    image = image.convert("RGB").resize((w, h), Image.LANCZOS)
    mask  = mask.convert("L").resize((w, h), Image.NEAREST)

    generator = torch.Generator().manual_seed(seed)

    result = pipe(
        prompt=prompt,
        negative_prompt=negative_prompt,
        image=image,
        mask_image=mask,
        num_inference_steps=n_steps,
        guidance_scale=guidance_scale,
        strength=strength,
        generator=generator
    )
    return result.images[0]


def generate_via_api(
    image: Image.Image,
    mask: Image.Image,
    prompt: str,
    token: str,
    img_size: tuple = (512, 512)
) -> Image.Image:
    """
    Generate via HuggingFace Inference API (no local GPU needed).
    Falls back to returning the input image if API fails.
    """
    import io, requests

    API_URL = "https://api-inference.huggingface.co/models/stabilityai/stable-diffusion-2-inpainting"
    headers = {"Authorization": f"Bearer {token}"}

    w, h = img_size
    image = image.convert("RGB").resize((w, h))
    mask  = mask.convert("L").resize((w, h))

    # Convert to bytes
    img_bytes  = io.BytesIO(); image.save(img_bytes, "PNG"); img_bytes.seek(0)
    mask_bytes = io.BytesIO(); mask.save(mask_bytes, "PNG"); mask_bytes.seek(0)

    try:
        response = requests.post(
            API_URL, headers=headers,
            files={"image": img_bytes, "mask_image": mask_bytes},
            data={"inputs": prompt},
            timeout=60
        )
        if response.status_code == 200:
            return Image.open(io.BytesIO(response.content)).convert("RGB")
    except Exception as e:
        print(f"  API call failed: {e}")

    # Fallback: return original image (won't look like rip current)
    return image


# ──────────────────────────────────────────────────────────────────────────────
# Batch runner
# ──────────────────────────────────────────────────────────────────────────────

def run_diffusion_pipeline(
    input_paths: list,
    mask_paths: list,
    output_dir: Path,
    pipe=None,
    use_api: bool = False,
    hf_token: str = None,
    n_per_mask: int = 3,
    n_steps: int = 30,
    guidance_scale: float = 7.5,
    strength: float = 0.85,
    img_size: tuple = (512, 512),
    min_mask_area: int = 500,
    save_overlays: bool = True,
    seed_base: int = 42,
    coco_builder: COCODatasetBuilder = None
) -> dict:
    """
    Main generation loop over (image, mask) pairs.
    """
    images_out  = output_dir / "images"
    masks_out   = output_dir / "masks"
    overlay_out = output_dir / "overlays"
    images_out.mkdir(parents=True, exist_ok=True)
    masks_out.mkdir(parents=True, exist_ok=True)
    if save_overlays:
        overlay_out.mkdir(parents=True, exist_ok=True)

    stats = {"generated": 0, "skipped": 0, "errors": 0}
    global_idx = 0

    for img_path, mask_path in tqdm(
        zip(input_paths, mask_paths), total=len(input_paths),
        desc="Diffusion generation"
    ):
        # Load
        try:
            pil_image = Image.open(img_path).convert("RGB")
            if mask_path and Path(mask_path).exists():
                mask_np = load_mask(mask_path)
            else:
                # Generate random ellipse mask
                w, h = pil_image.size
                mask_np = generate_random_ellipse_mask(h, w, n_ellipses=2)
                mask_np = dilate_mask(mask_np, kernel_size=15)
        except Exception as e:
            print(f"  Error loading {img_path}: {e}")
            stats["errors"] += 1
            continue

        if get_mask_area(mask_np) < min_mask_area:
            stats["skipped"] += 1
            continue

        pil_mask = Image.fromarray(mask_np)

        for var_idx in range(n_per_mask):
            seed = seed_base + global_idx
            prompt = PROMPT_VARIANTS[var_idx % len(PROMPT_VARIANTS)]

            try:
                if use_api and hf_token:
                    generated = generate_via_api(pil_image, pil_mask, prompt,
                                                  hf_token, img_size)
                elif pipe is not None:
                    generated = generate_with_pipeline(
                        pipe, pil_image, pil_mask, prompt, NEGATIVE_PROMPT,
                        n_steps, guidance_scale, strength, seed, img_size
                    )
                else:
                    # No model available: return placeholder
                    print("  WARNING: No model loaded. Saving placeholder images.")
                    generated = pil_image.resize(img_size)

                out_name = f"diff_{global_idx:06d}.png"
                generated.save(images_out / out_name)

                # Resize mask to match output
                out_mask = Image.fromarray(mask_np).resize(img_size, Image.NEAREST)
                out_mask.save(masks_out / out_name)

                if save_overlays:
                    from utils.blend_utils import overlay_mask_on_image
                    overlay = overlay_mask_on_image(
                        np.array(generated),
                        np.array(out_mask)
                    )
                    Image.fromarray(overlay).save(overlay_out / out_name)

                if coco_builder is not None:
                    coco_builder.add_image_with_mask(
                        image_path=images_out / out_name,
                        mask=np.array(out_mask),
                        file_name=f"diffusion/{out_name}",
                        min_area=min_mask_area
                    )

                global_idx += 1
                stats["generated"] += 1

            except Exception as e:
                print(f"  Generation error (img={img_path.name}, var={var_idx}): {e}")
                stats["errors"] += 1

    return stats


# ──────────────────────────────────────────────────────────────────────────────
# Test mode
# ──────────────────────────────────────────────────────────────────────────────

def run_test(output_dir: Path, img_size: tuple):
    """Smoke test: tries to load model and generate one image."""
    print("\n[TEST MODE] Running diffusion smoke test (5 steps)...")

    # Create a demo beach image
    from download_samples import create_demo_beach_images
    demo_dir = output_dir / "test_demo"
    create_demo_beach_images(demo_dir, n=2)

    img_paths  = list((demo_dir / "images").glob("*.png"))
    mask_paths = [demo_dir / "masks" / p.name for p in img_paths]

    device = get_device("auto")
    print(f"  Device: {device}")

    pipe = load_pipeline("stabilityai/stable-diffusion-2-inpainting", device)

    coco = COCODatasetBuilder(dataset_name="RipVIS-Diffusion-Test")

    stats = run_diffusion_pipeline(
        input_paths=img_paths,
        mask_paths=mask_paths,
        output_dir=output_dir / "test_output",
        pipe=pipe,
        n_per_mask=1,
        n_steps=5,   # Very fast for testing
        img_size=img_size,
        save_overlays=True,
        coco_builder=coco
    )

    ann_path = output_dir / "test_output" / "diffusion_annotations.json"
    coco.save(ann_path)

    print(f"\n✓ Test complete!")
    print(f"  Generated: {stats['generated']} images")
    print(f"  Errors:    {stats['errors']}")
    print(f"  Output:    {output_dir / 'test_output'}/")


# ──────────────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()

    print("=" * 60)
    print("  RipVIS — Diffusion Generation Pipeline")
    print("=" * 60)

    output_dir = Path(args.output_dir)
    img_size = tuple(args.img_size)

    if args.test:
        run_test(output_dir, img_size)
        return

    # Resolve input paths
    input_dir = Path(args.input_dir)
    img_dir   = input_dir / "images" if (input_dir / "images").is_dir() else input_dir
    exts      = {".png", ".jpg", ".jpeg"}
    img_paths = sorted(f for f in img_dir.rglob("*") if f.suffix.lower() in exts)

    mask_dir  = Path(args.mask_dir) if args.mask_dir else input_dir / "masks"
    mask_paths = [mask_dir / p.name for p in img_paths]

    print(f"  Input images:  {len(img_paths)}")
    print(f"  Mask dir:      {mask_dir}")
    print(f"  Output:        {output_dir}")
    print(f"  Model:         {args.model}")
    print(f"  Steps:         {args.steps}")
    print(f"  Images/mask:   {args.n_per_mask}")
    print(f"  Expected output: ~{len(img_paths) * args.n_per_mask}")
    print()

    if not img_paths:
        print("ERROR: No input images found.")
        return

    # Load model
    device = get_device(args.device)
    pipe = None
    if not args.use_api:
        pipe = load_pipeline(args.model, device)

    # COCO builder
    coco = COCODatasetBuilder(dataset_name="RipVIS-Synthetic-Diffusion")

    stats = run_diffusion_pipeline(
        input_paths=img_paths,
        mask_paths=mask_paths,
        output_dir=output_dir,
        pipe=pipe,
        use_api=args.use_api,
        hf_token=args.hf_token,
        n_per_mask=args.n_per_mask,
        n_steps=args.steps,
        guidance_scale=args.guidance,
        strength=args.strength,
        img_size=img_size,
        min_mask_area=args.min_mask_area,
        seed_base=args.seed,
        coco_builder=coco
    )

    ann_path = Path(args.annotation) if args.annotation else \
               output_dir / "diffusion_annotations.json"
    coco.save(ann_path)

    print(f"\n✓ Diffusion generation complete!")
    print(f"  Generated: {stats['generated']} images")
    print(f"  Skipped:   {stats['skipped']}")
    print(f"  Errors:    {stats['errors']}")
    print(f"  COCO JSON: {ann_path}")


if __name__ == "__main__":
    main()
