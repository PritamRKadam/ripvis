#!/usr/bin/env python3
"""
generate_five_swimmers.py
─────────────────────────
Generates 5 distinct scenarios with clearly visible swimmers in calm,
low-turbulence water using AnimateDiff + fine-tuned Water LoRA.
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
logger = logging.getLogger("CalmSwimmers")

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
    logger.info("Loading MotionAdapter...")
    adapter = MotionAdapter.from_pretrained("guoyww/animatediff-motion-adapter-v1-5-2", torch_dtype=torch.float16)
    
    logger.info("Loading SD 1.5 Pipeline...")
    pipe = AnimateDiffPipeline.from_pretrained("runwayml/stable-diffusion-v1-5", motion_adapter=adapter, torch_dtype=torch.float16)
    pipe.scheduler = DDIMScheduler.from_config(
        pipe.scheduler.config,
        beta_schedule="linear",
        clip_sample=False,
        timestep_spacing="linspace",
        steps_offset=1,
    )
    
    lora_dir = "models/water_swimmer_lora/final_lora"
    logger.info(f"Merging LoRA from {lora_dir}...")
    peft_unet = PeftModel.from_pretrained(pipe.unet, lora_dir)
    pipe.unet = peft_unet.merge_and_unload()
    pipe.to(device)
    pipe.enable_vae_slicing()

    out_dir = Path("runs/stage3_video_generation_v2")
    out_dir.mkdir(parents=True, exist_ok=True)

    neg_prompt = "(empty water:1.6), (no people:1.6), (no swimmers:1.6), empty ocean, uninhabited sea, nobody, violent waves, storm, rough sea, extreme turbulence, raging white foam, breaking surf, crashing waves, cartoon, 3d render, illustration, bad anatomy, deformed limbs, blurry, low quality"

    scenarios = [
        {
            "id": "scenario_01_breaststroke_calm_water",
            "title": "Swimmer in Calm Coastal Water Doing Breaststroke",
            "prompt": "(a woman swimming breaststroke in the water:1.5), (visible human swimmer:1.4), top-down aerial drone shot of a person swimming in calm clear blue ocean, delicate circular ripples, clear calm water surface, swimming motion, serene sea",
            "seed": 404,
        },
        {
            "id": "scenario_02_floating_supine_calm_sea",
            "title": "Person Floating Supine on Calm Ocean Surface",
            "prompt": "(a person floating peacefully on back:1.5), (visible human swimmer floating:1.4), top-down aerial drone shot of a person floating on calm blue ocean water, arms and legs outstretched, delicate water ripples, calm water surface, serene crystal clear sea",
            "seed": 777,
        },
        {
            "id": "scenario_03_freestyle_swimmer_calm_ocean",
            "title": "Freestyle Swimmer in Calm Coastal Waters",
            "prompt": "(an athletic swimmer swimming freestyle:1.5), (visible human swimmer:1.4), top-down aerial drone view of a person swimming through calm clear blue coastal water, rhythmic arm strokes and gentle flutter kicks, subtle trailing ripples, calm sea, photorealistic",
            "seed": 888,
        },
        {
            "id": "scenario_04_treading_water_coastal_channel",
            "title": "Swimmer Treading Water in Calm Coastal Current",
            "prompt": "(a person treading water:1.5), (visible human swimmer:1.4), overhead drone shot of a swimmer treading water in calm coastal ocean channel, head and shoulders above water, gentle undulating ripples, clear blue sea surface, peaceful calm water",
            "seed": 999,
        },
        {
            "id": "scenario_05_swimmer_gliding_gentle_ripples",
            "title": "Swimmer Gliding with Gentle Ripples",
            "prompt": "(a person swimming in calm water:1.5), (visible swimmer in ocean:1.4), top-down aerial drone view of a swimmer gliding across calm clear coastal water, gentle expanding ripple rings, serene ocean surface, natural fluid dynamics, realistic swimmer",
            "seed": 42,
        },
    ]

    import json
    summary_records = []

    for idx, scen in enumerate(scenarios):
        scen_id = scen["id"]
        logger.info(f"\n[{idx+1}/{len(scenarios)}] Generating: {scen['title']} (Seed: {scen['seed']})")
        
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
        scen_dir.mkdir(parents=True, exist_ok=True)

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

        summary_records.append({
            "id": scen_id,
            "title": scen["title"],
            "seed": scen["seed"],
            "prompt": scen["prompt"],
            "mp4": str(mp4_path),
            "webp": str(webp_path),
            "keyframes": str(sheet_path),
        })
        logger.info(f"✓ Completed {scen_id}")

    with open(out_dir / "generation_summary.json", "w") as f:
        json.dump(summary_records, f, indent=2)

    logger.info(f"\n🎉 All 5 calm swimmer video scenarios successfully generated in {out_dir}!")

if __name__ == "__main__":
    main()
