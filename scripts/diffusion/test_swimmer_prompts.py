#!/usr/bin/env python3
"""
test_swimmer_prompts.py
─────────────────────────
Diagnostic script to test prompt formulations, negative prompts, and LoRA scales
to ensure swimmers are distinctly visible in calm/low-turbulence water.
"""

import logging
import sys
from pathlib import Path
import torch
from PIL import Image
from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("TestSwimmer")

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Loading SD 1.5 on {device}...")
    pipe = StableDiffusionPipeline.from_pretrained(
        "runwayml/stable-diffusion-v1-5",
        torch_dtype=torch.float16 if device.type == "cuda" else torch.float32,
        safety_checker=None,
    )
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)

    # Attach Water LoRA
    lora_dir = "models/water_swimmer_lora/final_lora"
    if Path(lora_dir).exists():
        from peft import PeftModel
        peft_unet = PeftModel.from_pretrained(pipe.unet, lora_dir)
        pipe.unet = peft_unet.merge_and_unload()
        logger.info("Water LoRA attached successfully.")

    pipe.to(device)

    out_dir = Path("runs/prompt_tuning_tests")
    out_dir.mkdir(parents=True, exist_ok=True)

    test_configs = [
        {
            "id": "test_01_strong_swimmer_calm",
            "prompt": "(a human swimmer in the water:1.4), (person swimming freestyle:1.3), top-down aerial drone view of a visible person swimming in calm turquoise ocean water, gentle water ripples, subtle wake, calm sea surface, clear water, detailed human body with swim trunks",
            "negative": "(empty water:1.5), (no people:1.5), (no swimmers:1.5), empty sea, uninhabited ocean, violent waves, storm, rough turbulent water, excessive white foam, breaking waves, cartoon, blurry, bad anatomy",
            "seed": 101,
        },
        {
            "id": "test_02_floating_treading_calm",
            "prompt": "(a person floating on water surface:1.4), (swimmer treading water:1.3), top-down aerial drone view of a person floating peacefully in calm coastal ocean, subtle water ripples, sun caustics on shallow sea, crystal clear water, visible swimmer",
            "negative": "(empty water:1.5), (no people:1.5), (no swimmers:1.5), empty ocean, uninhabited water, violent waves, storm, rough water, turbulent foam, crashing surf, cartoon, blurry",
            "seed": 202,
        },
        {
            "id": "test_03_rip_swimmer_low_turbulence",
            "prompt": "(aerial view of a lone swimmer:1.4), (person swimming:1.3), top-down drone perspective of a human swimmer in calm ocean current channel, gentle water surface, subtle ripples, clear water, visible arms and head, natural coastal water",
            "negative": "(empty water:1.5), (no people:1.5), (no swimmers:1.5), uninhabited sea, violent foam, raging waves, stormy ocean, extreme turbulence, cartoon, blurry, deformed limbs",
            "seed": 303,
        },
        {
            "id": "test_04_breaststroke_calm",
            "prompt": "(a woman swimming breaststroke:1.4), (visible human swimmer:1.3), top-down aerial drone shot of a person swimming in clear calm blue water, delicate circular ripples, clear water surface, visible swimming motion, serene ocean",
            "negative": "(empty water:1.5), (no people:1.5), (no swimmers:1.5), empty sea, violent waves, white water, raging foam, high turbulence, cartoon, blurry",
            "seed": 404,
        },
    ]

    images = []
    for cfg in test_configs:
        logger.info(f"Generating {cfg['id']}...")
        gen = torch.Generator(device=device).manual_seed(cfg["seed"])
        img = pipe(
            prompt=cfg["prompt"],
            negative_prompt=cfg["negative"],
            num_inference_steps=25,
            guidance_scale=8.0,
            generator=gen,
            height=512,
            width=512,
        ).images[0]
        p = out_dir / f"{cfg['id']}.png"
        img.save(p)
        images.append(img)
        logger.info(f"Saved {p}")

    # Build 2x2 grid
    grid = Image.new("RGB", (1024, 1024))
    grid.paste(images[0], (0, 0))
    grid.paste(images[1], (512, 0))
    grid.paste(images[2], (0, 512))
    grid.paste(images[3], (512, 512))
    grid_path = out_dir / "test_swimmer_grid.png"
    grid.save(grid_path)
    logger.info(f"Saved test grid: {grid_path}")

if __name__ == "__main__":
    main()
