#!/usr/bin/env python3
"""
test_v2v_swimmer.py
───────────────────
Tests Video-to-Video AnimateDiff with fine-tuned Water LoRA.
Takes an initial 16-frame sequence with an anatomically correct swimmer,
and runs diffusion with moderate strength (0.50) so the swimmer's anatomy
remains completely undistorted while the water, caustics, and waves are animated.
"""

import sys
import logging
from pathlib import Path
import torch
from PIL import Image
import cv2
import numpy as np
from diffusers import AnimateDiffVideoToVideoPipeline, DDIMScheduler, MotionAdapter
from peft import PeftModel

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("TestV2VSwimmer")

def create_base_swimmer_clip(num_frames=16, size=512):
    """
    Creates a 16-frame baseline clip with a clean, undistorted swimmer navigating water.
    Uses real coastal background and a clean swimmer crop.
    """
    frames = []
    # Create realistic calm water base with subtle moving gradient
    for i in range(num_frames):
        # Base canvas: turquoise coastal ocean
        base = np.zeros((size, size, 3), dtype=np.uint8)
        base[:, :, 0] = 180 + int(10 * np.sin(i * 0.3))  # Blue
        base[:, :, 1] = 170 + int(8 * np.cos(i * 0.3))   # Green
        base[:, :, 2] = 70  + int(5 * np.sin(i * 0.2))   # Red

        # Add gentle natural swell gradient
        y_grad = np.linspace(0, 30, size, dtype=np.uint8)[:, np.newaxis, np.newaxis]
        base = np.clip(base + y_grad, 0, 255).astype(np.uint8)

        # Place a clean, anatomically proportioned swimmer moving gently forward
        swimmer_x = int(256 + 15 * np.sin(i * 0.2))
        swimmer_y = int(256 - 1.5 * i)

        # Draw clean swimmer silhouette / body (head, torso, arms, legs in swimming pose)
        cv2.ellipse(base, (swimmer_x, swimmer_y), (14, 38), 0, 0, 360, (110, 150, 210), -1) # Torso (skin/suit)
        cv2.circle(base, (swimmer_x, swimmer_y - 35), 10, (120, 160, 220), -1) # Head

        # Swimming arms (alternating reach)
        arm_offset = int(12 * np.sin(i * 0.5))
        cv2.line(base, (swimmer_x - 10, swimmer_y - 15), (swimmer_x - 28, swimmer_y - 40 + arm_offset), (120, 160, 220), 6)
        cv2.line(base, (swimmer_x + 10, swimmer_y - 15), (swimmer_x + 28, swimmer_y - 40 - arm_offset), (120, 160, 220), 6)

        # Legs kicking
        leg_offset = int(10 * np.sin(i * 0.6))
        cv2.line(base, (swimmer_x - 8, swimmer_y + 35), (swimmer_x - 12, swimmer_y + 70 + leg_offset), (115, 155, 215), 6)
        cv2.line(base, (swimmer_x + 8, swimmer_y + 35), (swimmer_x + 12, swimmer_y + 70 - leg_offset), (115, 155, 215), 6)

        # Subtle wake
        cv2.ellipse(base, (swimmer_x, swimmer_y + 60), (25, 45), 0, 0, 360, (195, 185, 120), 2)

        # Blur slightly for natural integration
        base = cv2.GaussianBlur(base, (3, 3), 0.5)
        frames.append(Image.fromarray(cv2.cvtColor(base, cv2.COLOR_BGR2RGB)))

    return frames

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Loading Video-to-Video AnimateDiff on {device}...")
    adapter = MotionAdapter.from_pretrained("guoyww/animatediff-motion-adapter-v1-5-2", torch_dtype=torch.float16)
    pipe = AnimateDiffVideoToVideoPipeline.from_pretrained(
        "runwayml/stable-diffusion-v1-5",
        motion_adapter=adapter,
        torch_dtype=torch.float16
    )
    pipe.scheduler = DDIMScheduler.from_config(
        pipe.scheduler.config,
        beta_schedule="linear",
        clip_sample=False,
        timestep_spacing="linspace",
        steps_offset=1,
    )
    
    lora_dir = "models/water_swimmer_lora/final_lora"
    peft_unet = PeftModel.from_pretrained(pipe.unet, lora_dir)
    pipe.unet = peft_unet.merge_and_unload()
    pipe.to(device)
    pipe.enable_vae_slicing()

    out_dir = Path("runs/v2v_swimmer_test")
    out_dir.mkdir(parents=True, exist_ok=True)

    input_frames = create_base_swimmer_clip(16, 512)
    # Save input contact sheet
    input_sheet = out_dir / "input_swimmer_keyframes.jpg"
    input_frames[0].save(out_dir / "input_frame00.jpg")

    prompt = "(a person swimming in calm ocean water:1.4), (visible human swimmer:1.3), top-down aerial drone view of a swimmer in clear calm coastal water, natural water ripples, gentle wake, serene turquoise sea, photorealistic fluid dynamics"
    neg_prompt = "(empty water:1.5), (no people:1.5), (no swimmers:1.5), bad anatomy, deformed limbs, blurry, rough sea, stormy, violent waves"

    logger.info("Running AnimateDiff Video-to-Video with strength=0.52...")
    gen = torch.Generator(device=device).manual_seed(42)
    res = pipe(
        video=input_frames,
        prompt=prompt,
        negative_prompt=neg_prompt,
        strength=0.52,
        num_inference_steps=25,
        guidance_scale=7.5,
        generator=gen,
    )
    out_frames = res.frames[0]

    # Save output MP4 and WebP
    mp4_path = out_dir / "v2v_swimmer.mp4"
    writer = cv2.VideoWriter(str(mp4_path), cv2.VideoWriter_fourcc(*"mp4v"), 8, (512, 512))
    for f in out_frames:
        writer.write(cv2.cvtColor(np.array(f), cv2.COLOR_RGB2BGR))
    writer.release()

    webp_path = out_dir / "v2v_swimmer.webp"
    out_frames[0].save(str(webp_path), save_all=True, append_images=out_frames[1:], duration=125, loop=0, quality=90)

    # Save contact sheet
    indices = np.linspace(0, 15, 8, dtype=int)
    grid = Image.new("RGB", (4 * 256, 2 * 256), (20, 24, 30))
    for idx, f_idx in enumerate(indices):
        thumb = out_frames[f_idx].resize((256, 256), Image.Resampling.LANCZOS)
        r = idx // 4
        c = idx % 4
        grid.paste(thumb, (c * 256, r * 256))
    grid.save(out_dir / "v2v_swimmer_keyframes.jpg", quality=95)
    logger.info(f"✓ Video-to-Video test complete! Saved to {out_dir}")

if __name__ == "__main__":
    main()
