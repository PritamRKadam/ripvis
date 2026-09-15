#!/usr/bin/env python3
"""
test_animatediff_swimmers.py
──────────────────────────────
Tests prompt formulations and LoRA vs non-LoRA for AnimateDiff to guarantee
prominent, clearly visible swimmers in calm water with minimal turbulence.
"""

import sys
import logging
from pathlib import Path
import torch
from PIL import Image
import cv2
import numpy as np
from diffusers import AnimateDiffPipeline, DDIMScheduler, MotionAdapter

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("TestAnimateDiffSwimmers")

def export_contact_sheet(frames, sheet_path, num_cols=4):
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
        thumb_cv = cv2.cvtColor(np.array(thumb), cv2.COLOR_RGB2BGR)
        cv2.putText(thumb_cv, f"Frame {frame_num:02d}", (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
        thumb_labeled = Image.fromarray(cv2.cvtColor(thumb_cv, cv2.COLOR_BGR2RGB))
        grid.paste(thumb_labeled, (c * thumb_w, r * thumb_h))
    sheet_path.parent.mkdir(parents=True, exist_ok=True)
    grid.save(str(sheet_path), quality=95)

def export_video_cv2(frames, output_path, fps=8):
    w, h = frames[0].size
    output_path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(output_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    for frame in frames:
        writer.write(cv2.cvtColor(np.array(frame), cv2.COLOR_RGB2BGR))
    writer.release()

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    adapter = MotionAdapter.from_pretrained("guoyww/animatediff-motion-adapter-v1-5-2", torch_dtype=torch.float16)
    pipe = AnimateDiffPipeline.from_pretrained("runwayml/stable-diffusion-v1-5", motion_adapter=adapter, torch_dtype=torch.float16)
    pipe.scheduler = DDIMScheduler.from_config(
        pipe.scheduler.config,
        beta_schedule="linear",
        clip_sample=False,
        timestep_spacing="linspace",
        steps_offset=1,
    )
    
    # Check with LoRA
    lora_dir = "models/water_swimmer_lora/final_lora"
    from peft import PeftModel
    peft_unet = PeftModel.from_pretrained(pipe.unet, lora_dir)
    pipe.unet = peft_unet.merge_and_unload()
    pipe.to(device)
    pipe.enable_vae_slicing()

    out_dir = Path("runs/animatediff_swimmer_tests")
    out_dir.mkdir(parents=True, exist_ok=True)

    neg_prompt = "(empty water:1.6), (no people:1.6), (no swimmers:1.6), empty ocean, uninhabited sea, nobody, violent waves, storm, rough sea, extreme turbulence, raging white foam, breaking surf, crashing waves, cartoon, 3d render, bad anatomy, deformed limbs, blurry, low quality"

    tests = [
        {
            "id": "anim_breaststroke_calm",
            "prompt": "(a woman swimming breaststroke in the water:1.5), (visible human swimmer:1.4), top-down aerial drone shot of a person swimming in calm clear blue ocean, delicate circular ripples, clear calm water surface, swimming motion, serene sea",
            "seed": 404,
        },
        {
            "id": "anim_freestyle_swimmer_calm",
            "prompt": "(a person swimming freestyle:1.5), (human swimmer in water:1.4), top-down aerial drone view of a swimmer in calm turquoise sea, swimming forward with arm strokes and flutter kicks, subtle trailing wake, gentle water ripples, calm water surface",
            "seed": 101,
        },
        {
            "id": "anim_floating_treading_calm",
            "prompt": "(a person floating on water:1.5), (swimmer in ocean:1.4), high-angle drone view of a visible person floating peacefully on calm turquoise water surface, gentle ripples, serene calm sea, clear water, visible swimming attire",
            "seed": 202,
        },
    ]

    for t in tests:
        logger.info(f"Generating AnimateDiff video: {t['id']}...")
        gen = torch.Generator(device=device).manual_seed(t["seed"])
        res = pipe(
            prompt=t["prompt"],
            negative_prompt=neg_prompt,
            num_frames=16,
            height=512,
            width=512,
            num_inference_steps=25,
            guidance_scale=8.0,
            generator=gen,
        )
        frames = res.frames[0]
        sheet_path = out_dir / f"{t['id']}_keyframes.jpg"
        mp4_path = out_dir / f"{t['id']}.mp4"
        webp_path = out_dir / f"{t['id']}.webp"
        export_contact_sheet(frames, sheet_path)
        export_video_cv2(frames, mp4_path)
        frames[0].save(str(webp_path), save_all=True, append_images=frames[1:], duration=125, loop=0, quality=90)
        logger.info(f"Exported {sheet_path} and {mp4_path}")

if __name__ == "__main__":
    main()
