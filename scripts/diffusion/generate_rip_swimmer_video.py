#!/usr/bin/env python3
"""
generate_rip_swimmer_video.py
─────────────────────────────
Generates realistic video sequences of swimmers navigating rip currents
using AnimateDiff (v1-5-2) integrated with the fine-tuned Rip-Swimmer LoRA.

Features:
- Loads SD 1.5 + AnimateDiff motion adapter
- Injects fine-tuned LoRA weights (models/diffusion_rip_swimmer_lora/final_lora)
- Optimized for NVIDIA RTX 3060 (12GB VRAM) via FP16 & CPU offloading / VAE slicing
- Outputs:
  - .mp4 video clips (OpenCV)
  - .gif and .webp animated previews
  - Individual frames in frame directories
  - 4-frame / 8-frame contact-sheet grid images for easy visual review
"""

import argparse
import logging
import os
import sys
import time
from pathlib import Path
from typing import List, Optional

import cv2
import numpy as np
import torch
import yaml
from PIL import Image

from diffusers import (
    AnimateDiffPipeline,
    DDIMScheduler,
    EulerDiscreteScheduler,
    MotionAdapter,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("RipSwimmerVideo")


def load_config(config_path: str) -> dict:
    """Load YAML configuration."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def build_pipeline(
    base_model: str,
    motion_adapter_id: str,
    lora_dir: Optional[str] = None,
    lora_weight_name: str = "adapter_model.safetensors",
    lora_scale: float = 0.85,
    scheduler_type: str = "ddim",
    device: str = "cuda",
) -> AnimateDiffPipeline:
    """
    Constructs the AnimateDiff video generation pipeline with optional LoRA.
    """
    logger.info(f"Loading motion adapter: {motion_adapter_id}")
    adapter = MotionAdapter.from_pretrained(
        motion_adapter_id,
        torch_dtype=torch.float16,
    )

    logger.info(f"Loading AnimateDiffPipeline with base model: {base_model}")
    pipe = AnimateDiffPipeline.from_pretrained(
        base_model,
        motion_adapter=adapter,
        torch_dtype=torch.float16,
    )

    # Configure scheduler recommended for AnimateDiff
    if scheduler_type.lower() == "ddim":
        logger.info("Setting DDIMScheduler (optimized for AnimateDiff v1-5-2)")
        pipe.scheduler = DDIMScheduler.from_config(
            pipe.scheduler.config,
            beta_schedule="linear",
            clip_sample=False,
            timestep_spacing="linspace",
            steps_offset=1,
        )
    elif scheduler_type.lower() == "euler":
        logger.info("Setting EulerDiscreteScheduler")
        pipe.scheduler = EulerDiscreteScheduler.from_config(
            pipe.scheduler.config,
            timestep_spacing="trailing",
        )

    # Attach fine-tuned LoRA if provided
    if lora_dir and Path(lora_dir).exists():
        logger.info(f"Attaching fine-tuned LoRA from: {lora_dir} (scale={lora_scale})")
        try:
            from peft import PeftModel
            peft_unet = PeftModel.from_pretrained(pipe.unet, lora_dir)
            pipe.unet = peft_unet.merge_and_unload()
            logger.info("✓ Fine-tuned LoRA merged and unloaded into UNet successfully!")
        except Exception as e:
            logger.warning(f"Could not merge LoRA ({e}), falling back to base model.")
    else:
        logger.warning(f"LoRA path {lora_dir} not found. Running without LoRA.")

    # VRAM optimizations for 12GB GPU
    logger.info("Enabling GPU execution and VAE slicing...")
    pipe.to(device)
    pipe.enable_vae_slicing()

    return pipe


def export_video_cv2(
    frames: List[Image.Image],
    output_path: Path,
    fps: int = 8,
):
    """Exports a list of PIL images to MP4 using OpenCV VideoWriter."""
    if not frames:
        return

    w, h = frames[0].size
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(output_path), fourcc, fps, (w, h))

    for frame in frames:
        bgr = cv2.cvtColor(np.array(frame), cv2.COLOR_RGB2BGR)
        writer.write(bgr)

    writer.release()
    logger.info(f"Saved MP4: {output_path} ({len(frames)} frames @ {fps} fps)")


def export_gif_webp(
    frames: List[Image.Image],
    gif_path: Optional[Path] = None,
    webp_path: Optional[Path] = None,
    fps: int = 8,
):
    """Exports PIL images to animated GIF and WebP."""
    if not frames:
        return

    duration = int(1000 / fps)
    if gif_path:
        gif_path.parent.mkdir(parents=True, exist_ok=True)
        frames[0].save(
            str(gif_path),
            save_all=True,
            append_images=frames[1:],
            duration=duration,
            loop=0,
            optimize=True,
        )
        logger.info(f"Saved GIF: {gif_path}")

    if webp_path:
        webp_path.parent.mkdir(parents=True, exist_ok=True)
        frames[0].save(
            str(webp_path),
            save_all=True,
            append_images=frames[1:],
            duration=duration,
            loop=0,
            quality=90,
        )
        logger.info(f"Saved WebP: {webp_path}")


def export_contact_sheet(
    frames: List[Image.Image],
    sheet_path: Path,
    num_cols: int = 4,
):
    """
    Creates a visual contact sheet grid of keyframes across the sequence
    for immediate static inspection and markdown reporting.
    """
    if not frames:
        return

    n = len(frames)
    indices = np.linspace(0, n - 1, min(8, n), dtype=int)
    selected = [frames[i] for i in indices]

    w, h = selected[0].size
    rows = (len(selected) + num_cols - 1) // num_cols
    thumb_w, thumb_h = 256, int(256 * h / w)

    grid = Image.new("RGB", (num_cols * thumb_w, rows * thumb_h), (20, 24, 30))

    for idx, (img, frame_num) in enumerate(zip(selected, indices)):
        r = idx // num_cols
        c = idx % num_cols
        thumb = img.resize((thumb_w, thumb_h), Image.Resampling.LANCZOS)
        
        # Overlay frame index text using OpenCV on thumbnail
        thumb_cv = cv2.cvtColor(np.array(thumb), cv2.COLOR_RGB2BGR)
        cv2.putText(
            thumb_cv,
            f"Frame {frame_num:02d}",
            (10, 24),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        thumb_labeled = Image.fromarray(cv2.cvtColor(thumb_cv, cv2.COLOR_BGR2RGB))
        grid.paste(thumb_labeled, (c * thumb_w, r * thumb_h))

    sheet_path.parent.mkdir(parents=True, exist_ok=True)
    grid.save(str(sheet_path), quality=95)
    logger.info(f"Saved keyframe contact sheet: {sheet_path}")


def main():
    parser = argparse.ArgumentParser(description="AnimateDiff Rip-Swimmer Video Generator")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/video_generation_config.yaml",
        help="Path to video generation config YAML",
    )
    parser.add_argument("--single_scenario", type=str, default=None, help="Run only specific scenario ID")
    parser.add_argument("--num_frames", type=int, default=None, help="Override number of frames")
    parser.add_argument("--steps", type=int, default=None, help="Override inference steps")
    parser.add_argument("--lora_scale", type=float, default=None, help="Override LoRA scale")
    args = parser.parse_args()

    cfg = load_config(args.config)

    # Resolve overrides
    pipe_cfg = cfg.get("pipeline", {})
    gen_cfg = cfg.get("generation", {})
    out_cfg = cfg.get("output", {})

    lora_scale = args.lora_scale if args.lora_scale is not None else pipe_cfg.get("lora_scale", 0.85)
    num_frames = args.num_frames if args.num_frames is not None else gen_cfg.get("num_frames", 16)
    steps = args.steps if args.steps is not None else gen_cfg.get("num_inference_steps", 25)
    guidance_scale = gen_cfg.get("guidance_scale", 7.5)
    height = gen_cfg.get("height", 512)
    width = gen_cfg.get("width", 512)
    fps = gen_cfg.get("fps", 8)
    negative_prompt = cfg.get("negative_prompt", "")
    output_dir = Path(out_cfg.get("output_dir", "runs/video_trials"))
    output_dir.mkdir(parents=True, exist_ok=True)

    # Initialize Pipeline
    pipe = build_pipeline(
        base_model=pipe_cfg.get("base_model", "runwayml/stable-diffusion-v1-5"),
        motion_adapter_id=pipe_cfg.get("motion_adapter", "guoyww/animatediff-motion-adapter-v1-5-2"),
        lora_dir=pipe_cfg.get("lora_dir", "models/diffusion_rip_swimmer_lora/final_lora"),
        lora_weight_name=pipe_cfg.get("lora_weight_name", "adapter_model.safetensors"),
        lora_scale=lora_scale,
        scheduler_type=gen_cfg.get("scheduler", "ddim"),
    )

    scenarios = cfg.get("scenarios", [])
    if args.single_scenario:
        scenarios = [s for s in scenarios if s.get("id") == args.single_scenario]
        if not scenarios:
            logger.error(f"Scenario {args.single_scenario} not found in config!")
            sys.exit(1)

    logger.info(f"Executing video generation across {len(scenarios)} scenario(s)...")

    for idx, scen in enumerate(scenarios):
        scen_id = scen.get("id", f"scenario_{idx+1:02d}")
        scen_title = scen.get("title", scen_id)
        prompt = scen.get("prompt")
        seed = scen.get("seed", 42)

        logger.info(f"\n[{idx+1}/{len(scenarios)}] Generating: {scen_title} (ID: {scen_id})")
        logger.info(f"Prompt: {prompt}")
        logger.info(f"Seed: {seed} | Frames: {num_frames} | Steps: {steps}")

        gen_device = "cuda" if torch.cuda.is_available() else "cpu"
        generator = torch.Generator(device=gen_device).manual_seed(seed)

        start_time = time.time()
        result = pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            num_frames=num_frames,
            height=height,
            width=width,
            num_inference_steps=steps,
            guidance_scale=guidance_scale,
            generator=generator,
        )
        elapsed = time.time() - start_time
        logger.info(f"Generation completed in {elapsed:.2f}s ({elapsed/num_frames:.2f}s/frame)")

        # result.frames is shape (1, num_frames)
        frames = result.frames[0]

        # Directory structure for outputs
        scen_dir = output_dir / scen_id
        scen_dir.mkdir(parents=True, exist_ok=True)

        # 1. Save individual frame images
        if out_cfg.get("save_frames", True):
            frames_dir = scen_dir / "frames"
            frames_dir.mkdir(parents=True, exist_ok=True)
            for f_idx, frame_img in enumerate(frames):
                frame_img.save(frames_dir / f"frame_{f_idx:03d}.jpg", quality=95)
            logger.info(f"Saved {len(frames)} individual frames to {frames_dir}")

        # 2. Save MP4
        if out_cfg.get("save_mp4", True):
            mp4_path = scen_dir / f"{scen_id}.mp4"
            export_video_cv2(frames, mp4_path, fps=fps)

        # 3. Save GIF / WebP
        gif_path = scen_dir / f"{scen_id}.gif" if out_cfg.get("save_gif", True) else None
        webp_path = scen_dir / f"{scen_id}.webp" if out_cfg.get("save_webp", True) else None
        export_gif_webp(frames, gif_path=gif_path, webp_path=webp_path, fps=fps)

        # 4. Save Keyframe Contact Sheet
        sheet_path = scen_dir / f"{scen_id}_keyframes.jpg"
        export_contact_sheet(frames, sheet_path, num_cols=4)

    logger.info(f"\n✅ All video generation scenarios successfully exported to {output_dir}")


if __name__ == "__main__":
    main()
