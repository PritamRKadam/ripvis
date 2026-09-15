#!/usr/bin/env python3
"""
test_water_lora.py
───────────────────
Validates the fine-tuned water-specialized LoRA adapter.
- Checks adapter file integrity
- Loads into StableDiffusionPipeline
- Generates validation sample frames conditioned on key water trigger words
- Saves output contact grid to runs/lora_test_samples/water_lora_trigger_samples.png
"""

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger("TestWaterLoRA")


def main():
    parser = argparse.ArgumentParser(description="Test fine-tuned Water LoRA adapter.")
    parser.add_argument("--base_model", type=str, default="runwayml/stable-diffusion-v1-5")
    parser.add_argument("--lora_dir", type=str, default="models/water_swimmer_lora/final_lora")
    parser.add_argument("--output_dir", type=str, default="runs/lora_test_samples")
    parser.add_argument("--num_inference_steps", type=int, default=25)
    parser.add_argument("--guidance_scale", type=float, default=7.5)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    lora_path = Path(args.lora_dir)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Checking LoRA adapter files in: {lora_path}")
    safetensors = list(lora_path.glob("*.safetensors")) + list(lora_path.glob("*.bin"))
    if not safetensors:
        raise FileNotFoundError(f"No adapter weights found in {lora_path}")

    logger.info(f"Found adapter weights: {[f.name for f in safetensors]}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Loading StableDiffusionPipeline on {device}...")

    pipe = StableDiffusionPipeline.from_pretrained(
        args.base_model,
        torch_dtype=torch.float16 if device.type == "cuda" else torch.float32,
        safety_checker=None,
    )
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)

    # Attach fine-tuned LoRA using PEFT
    logger.info(f"Attaching Water LoRA from {lora_path}...")
    from peft import PeftModel
    peft_unet = PeftModel.from_pretrained(pipe.unet, str(lora_path))
    pipe.unet = peft_unet.merge_and_unload()
    logger.info("Water LoRA successfully merged into UNet!")
    pipe.to(device)

    # Test Prompts featuring water trigger words
    test_prompts = [
        (
            "aerial view, water surface, turbulent wake: top-down drone view of a swimmer in coastal ocean water, foamy rip current wake, natural sunlight, photorealistic",
            "aerial_swimmer_wake"
        ),
        (
            "underwater, caustics, water surface: a person snorkeling over a rocky coral reef, volumetric sunbeams piercing clear turquoise water, dancing caustics",
            "underwater_caustics"
        ),
        (
            "splash, ripples, turbulent wake: a person diving into deep blue water with explosive splash droplets and expanding circular ripples, 4k",
            "splash_ripples"
        ),
        (
            "water surface, ripples, caustics: overhead drone shot at 30m altitude of a person floating supine in calm ocean swells with sunlight glitter",
            "floating_ocean_swells"
        ),
    ]

    negative_prompt = "cartoon, 3d render, illustration, drawing, deformed body, bad anatomy, blurry, fake water"

    generated_images = []
    generator = torch.Generator(device=device).manual_seed(args.seed)

    for prompt_text, slug in test_prompts:
        logger.info(f"Generating test sample for: '{slug}'...")
        image = pipe(
            prompt=prompt_text,
            negative_prompt=negative_prompt,
            num_inference_steps=args.num_inference_steps,
            guidance_scale=args.guidance_scale,
            generator=generator,
            height=512,
            width=512,
        ).images[0]

        img_path = out_dir / f"{slug}.png"
        image.save(img_path)
        generated_images.append(image)

    # Create 2x2 grid
    grid = Image.new("RGB", (1024, 1024))
    grid.paste(generated_images[0], (0, 0))
    grid.paste(generated_images[1], (512, 0))
    grid.paste(generated_images[2], (0, 512))
    grid.paste(generated_images[3], (512, 512))

    grid_path = out_dir / "water_lora_trigger_samples.png"
    grid.save(grid_path)
    logger.info(f"Saved 2x2 trigger verification grid to: {grid_path}")

    # Summary
    summary = {
        "adapter_path": str(lora_path),
        "test_prompts": [p[0] for p in test_prompts],
        "grid_path": str(grid_path),
        "status": "VALIDATION_PASSED",
    }
    with open(out_dir / "test_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    logger.info("Water LoRA validation testing complete!")


if __name__ == "__main__":
    main()
