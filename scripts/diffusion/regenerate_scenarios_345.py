#!/usr/bin/env python3
"""
regenerate_scenarios_345.py
────────────────────────────
Refines scenarios 3, 4, and 5 with centered, persistent swimmer postures
so that all 5 scenarios feature unambiguous, clearly visible swimmers across all 16 frames.
"""

import logging
from pathlib import Path
import torch
from PIL import Image
import cv2
import numpy as np
from diffusers import AnimateDiffPipeline, DDIMScheduler, MotionAdapter
from peft import PeftModel

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("RefineSwimmers")

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
    
    lora_dir = "models/water_swimmer_lora/final_lora"
    peft_unet = PeftModel.from_pretrained(pipe.unet, lora_dir)
    pipe.unet = peft_unet.merge_and_unload()
    pipe.to(device)
    pipe.enable_vae_slicing()

    out_dir = Path("runs/stage3_video_generation_v2")

    neg_prompt = "(empty water:1.6), (no people:1.6), (no swimmers:1.6), empty ocean, uninhabited sea, nobody, violent waves, storm, rough sea, extreme turbulence, raging white foam, breaking surf, crashing waves, cartoon, 3d render, illustration, bad anatomy, deformed limbs, blurry, low quality"

    refined_scenarios = [
        {
            "id": "scenario_03_freestyle_swimmer_calm_ocean",
            "title": "Freestyle Swimmer in Calm Coastal Waters",
            "prompt": "(a woman swimming breaststroke in calm water:1.5), (visible human swimmer centered in ocean:1.4), top-down aerial drone view of a person swimming in clear calm blue coastal water, rhythmic swimming strokes, gentle expanding ripples, calm sea, centered view",
            "seed": 512,
        },
        {
            "id": "scenario_04_treading_water_coastal_channel",
            "title": "Swimmer Treading Water in Calm Coastal Current",
            "prompt": "(a person treading water in place:1.5), (visible swimmer centered in water:1.4), top-down aerial drone shot of a person treading water in calm clear coastal ocean, head and torso visible above calm water surface, gentle circular ripples, serene sea",
            "seed": 618,
        },
        {
            "id": "scenario_05_swimmer_gliding_gentle_ripples",
            "title": "Swimmer Floating with Swim Ring in Calm Water",
            "prompt": "(a person floating with bright yellow swim ring in calm water:1.5), (visible swimmer in ocean:1.4), top-down aerial drone shot of a person floating peacefully in calm blue sea water, gentle ripples, serene ocean surface, clear water, centered swimmer",
            "seed": 733,
        },
    ]

    for idx, scen in enumerate(refined_scenarios):
        scen_id = scen["id"]
        logger.info(f"\nRefining [{idx+1}/3]: {scen['title']} (Seed: {scen['seed']})")
        gen = torch.Generator(device=device).manual_seed(scen["seed"])
        res = pipe(
            prompt=scen["prompt"],
            negative_prompt=neg_prompt,
            num_frames=16,
            height=512,
            width=512,
            num_inference_steps=25,
            guidance_scale=8.0,
            generator=gen,
        )
        frames = res.frames[0]
        scen_dir = out_dir / scen_id

        # 1. Frames
        frames_dir = scen_dir / "frames"
        frames_dir.mkdir(parents=True, exist_ok=True)
        for f_idx, f_img in enumerate(frames):
            f_img.save(frames_dir / f"frame_{f_idx:03d}.jpg", quality=95)

        # 2. MP4
        mp4_path = scen_dir / f"{scen_id}.mp4"
        export_video_cv2(frames, mp4_path, fps=8)

        # 3. GIF and WebP
        gif_path = scen_dir / f"{scen_id}.gif"
        webp_path = scen_dir / f"{scen_id}.webp"
        frames[0].save(str(gif_path), save_all=True, append_images=frames[1:], duration=125, loop=0, optimize=True)
        frames[0].save(str(webp_path), save_all=True, append_images=frames[1:], duration=125, loop=0, quality=90)

        # 4. Contact sheet
        sheet_path = scen_dir / f"{scen_id}_keyframes.jpg"
        export_contact_sheet(frames, sheet_path)
        logger.info(f"✓ Refined {scen_id}")

if __name__ == "__main__":
    main()
