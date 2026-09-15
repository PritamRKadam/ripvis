#!/usr/bin/env python3
"""
add_swimmers_to_video.py
────────────────────────
Composites realistic swimmers with physical water immersion and hydrodynamic
drift into rip current video sequences.

Features:
- Uses specialized aerial swimmer actions (freestyle, breaststroke, treading, floating)
- Implements authentic seaward hydrodynamic drift trajectories along rip current conduits
- Applies Beer-Lambert optical depth absorption & subsurface refraction
- Adds wave disturbance / swell bobbing around the swimmer
- Outputs high-quality MP4, animated WebP/GIF, and side-by-side comparison videos
"""

import argparse
import glob
import math
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from utils.person_utils import blend_swimmer_physically, extract_clean_swimmer_cutout


def create_drift_trajectory(
    num_frames: int,
    start_pos: Tuple[float, float],
    end_pos: Tuple[float, float],
    wave_amplitude: float = 3.0,
    wave_frequency: float = 1.0,
) -> List[Tuple[int, int]]:
    """
    Generates a smooth seaward drift trajectory with subtle wave bobbing.
    """
    trajectory = []
    x0, y0 = start_pos
    x1, y1 = end_pos

    for i in range(num_frames):
        t = i / max(1, num_frames - 1)
        base_x = x0 + t * (x1 - x0)
        base_y = y0 + t * (y1 - y0)

        # Perpendicular wave bobbing / swell perturbation
        drift_angle = math.atan2(y1 - y0, x1 - x0)
        perp_angle = drift_angle + math.pi / 2.0
        bobbing = math.sin(t * 2 * math.pi * wave_frequency) * wave_amplitude

        curr_x = int(round(base_x + bobbing * math.cos(perp_angle)))
        curr_y = int(round(base_y + bobbing * math.sin(perp_angle)))
        trajectory.append((curr_x, curr_y))

    return trajectory


def composite_swimmer_into_video(
    frames_dir: Path,
    output_dir: Path,
    swimmer_rgba: np.ndarray,
    start_pos: Tuple[float, float],
    end_pos: Tuple[float, float],
    scale: float = 0.36,
    base_angle: float = 0.0,
    submerged_axis: str = "vertical",
    fps: int = 8,
    depth_intensity: float = 0.28,
    exposure_factor: float = 0.92,
    scenario_name: str = "rip_swimmer",
    title_label: str = "Rip Current Video + Harmonized Swimmer",
):
    """
    Composites swimmer across all frames and exports MP4, GIF, WebP, and comparisons.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    frame_files = sorted(frames_dir.glob("*.jpg")) + sorted(frames_dir.glob("*.png"))
    if not frame_files:
        raise ValueError(f"No frames found in {frames_dir}")

    num_frames = len(frame_files)
    trajectory = create_drift_trajectory(num_frames, start_pos, end_pos)

    orig_frames = []
    composited_frames = []
    sbs_frames = []

    h, w = 512, 512

    for idx, (fpath, (tx, ty)) in enumerate(zip(frame_files, trajectory)):
        frame_bgr = cv2.imread(str(fpath))
        if frame_bgr is None:
            continue
        h, w = frame_bgr.shape[:2]
        orig_frames.append(frame_bgr.copy())

        # Subtle dynamic swell breathing
        dynamic_scale = scale * (1.0 + 0.03 * math.sin(idx * 0.7))
        dynamic_angle = base_angle + 3.0 * math.sin(idx * 0.8)

        # Blend swimmer with Beer-Lambert physical depth attenuation
        canvas_with_swimmer, mask, bbox = blend_swimmer_physically(
            canvas=frame_bgr.copy(),
            swimmer_rgba=swimmer_rgba,
            target_xy=(tx, ty),
            scale=dynamic_scale,
            angle=dynamic_angle,
            submerged_axis=submerged_axis,
            depth_intensity=depth_intensity,
            exposure_factor=exposure_factor,
            color_format="BGR",
        )

        composited_frames.append(canvas_with_swimmer)

        # Side by side: Original Rip Video | With Harmonized Swimmer
        sbs = np.concatenate([frame_bgr, canvas_with_swimmer], axis=1)
        # Add informative label banners
        banner = np.zeros((36, sbs.shape[1], 3), dtype=np.uint8)
        cv2.putText(banner, "Rip Current Video (Original)", (25, 24),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (220, 220, 220), 2)
        cv2.putText(banner, title_label, (w + 25, 24),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (50, 220, 255), 2)
        sbs_with_banner = np.concatenate([banner, sbs], axis=0)
        sbs_frames.append(sbs_with_banner)

    # Export MP4 (Composited)
    mp4_out = output_dir / f"{scenario_name}_with_swimmer.mp4"
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(mp4_out), fourcc, fps, (w, h))
    for f in composited_frames:
        writer.write(f)
    writer.release()
    print(f"  Saved MP4: {mp4_out}")

    # Export MP4 (Side-by-Side)
    sbs_mp4_out = output_dir / f"{scenario_name}_comparison_sbs.mp4"
    sbs_h, sbs_w = sbs_frames[0].shape[:2]
    sbs_writer = cv2.VideoWriter(str(sbs_mp4_out), fourcc, fps, (sbs_w, sbs_h))
    for f in sbs_frames:
        sbs_writer.write(f)
    sbs_writer.release()
    print(f"  Saved Side-by-Side MP4: {sbs_mp4_out}")

    # Export Animated GIF & WebP
    pil_frames = [Image.fromarray(cv2.cvtColor(f, cv2.COLOR_BGR2RGB)) for f in composited_frames]
    duration_ms = int(1000 / fps)

    gif_out = output_dir / f"{scenario_name}_with_swimmer.gif"
    pil_frames[0].save(
        gif_out,
        save_all=True,
        append_images=pil_frames[1:],
        duration=duration_ms,
        loop=0,
        optimize=True,
    )
    print(f"  Saved GIF: {gif_out}")

    webp_out = output_dir / f"{scenario_name}_with_swimmer.webp"
    pil_frames[0].save(
        webp_out,
        save_all=True,
        append_images=pil_frames[1:],
        duration=duration_ms,
        loop=0,
        lossless=False,
        quality=88,
    )
    print(f"  Saved WebP: {webp_out}")

    # Export 4-Keyframe Preview Collage
    keyframe_indices = [0, num_frames // 3, 2 * num_frames // 3, num_frames - 1]
    key_patches = []
    for ki in keyframe_indices:
        kf = composited_frames[ki].copy()
        cv2.putText(kf, f"Frame {ki+1}/{num_frames}", (15, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        key_patches.append(kf)

    collage = np.concatenate(key_patches, axis=1)
    collage_path = output_dir / f"{scenario_name}_keyframes.jpg"
    cv2.imwrite(str(collage_path), collage)
    print(f"  Saved Keyframe Grid: {collage_path}")

    return mp4_out, sbs_mp4_out, gif_out, collage_path


def main():
    parser = argparse.ArgumentParser(description="Add Swimmers to Rip Current Videos")
    parser.add_argument(
        "--scenario",
        type=str,
        default="all",
        choices=["scenario_01", "scenario_02", "scenario_03", "scenario_04", "all"],
        help="Scenario ID to process (default: all)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="runs/video_trials_with_swimmers",
        help="Directory to save output videos",
    )
    args = parser.parse_args()

    crops_dir = Path("data/afo_samples/a_1044_crops")
    sw_freestyle = extract_clean_swimmer_cutout(crops_dir / "22_human_103x101.png", color_format="BGR")
    sw_breaststroke = extract_clean_swimmer_cutout(crops_dir / "26_human_128x99.png", color_format="BGR")
    sw_treading = extract_clean_swimmer_cutout(crops_dir / "46_human_135x162.png", color_format="BGR")
    sw_floating = extract_clean_swimmer_cutout(crops_dir / "30_human_151x134.png", color_format="BGR")

    scenarios = {
        "scenario_01": {
            "name": "scenario_01_rip_neck_swimmer",
            "title": "Freestyle Swimmer Drifting in Outgoing Rip Neck",
            "frames_dir": Path("runs/video_trials/scenario_01_rip_neck_swimmer/frames"),
            "swimmer_rgba": sw_freestyle,
            "start_pos": (260, 360),
            "end_pos": (220, 200),
            "scale": 0.36,
            "base_angle": -65.0,
            "submerged_axis": "horizontal",
            "depth_intensity": 0.28,
        },
        "scenario_02": {
            "name": "scenario_02_treading_near_waves",
            "title": "Swimmer Treading Water Near Breaking Surf",
            "frames_dir": Path("runs/video_trials/scenario_02_treading_near_waves/frames"),
            "swimmer_rgba": sw_treading,
            "start_pos": (255, 270),
            "end_pos": (245, 230),
            "scale": 0.32,
            "base_angle": -15.0,
            "submerged_axis": "vertical",
            "depth_intensity": 0.30,
        },
        "scenario_03": {
            "name": "scenario_03_freestyle_rip_plume",
            "title": "Breaststroke Swimmer in Outgoing Plume",
            "frames_dir": Path("runs/video_trials/scenario_03_freestyle_rip_plume/frames"),
            "swimmer_rgba": sw_breaststroke,
            "start_pos": (270, 340),
            "end_pos": (290, 180),
            "scale": 0.34,
            "base_angle": -75.0,
            "submerged_axis": "horizontal",
            "depth_intensity": 0.26,
        },
        "scenario_04": {
            "name": "scenario_04_rip_head_offshore",
            "title": "Floating Swimmer in Expanding Offshore Rip Head",
            "frames_dir": Path("runs/video_trials/scenario_04_rip_head_offshore/frames"),
            "swimmer_rgba": sw_floating,
            "start_pos": (250, 270),
            "end_pos": (220, 190),
            "scale": 0.33,
            "base_angle": -35.0,
            "submerged_axis": "vertical",
            "depth_intensity": 0.30,
        },
    }

    base_out = Path(args.output_dir)
    base_out.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("  Adding Photorealistic Harmonized Swimmers to Rip Current Videos")
    print("=" * 60)

    selected_scenarios = (
        scenarios.items()
        if args.scenario == "all"
        else [(args.scenario, scenarios[args.scenario])]
    )

    for sc_id, sc_cfg in selected_scenarios:
        print(f"\nProcessing {sc_id} ({sc_cfg['name']})...")
        if not sc_cfg["frames_dir"].exists():
            print(f"  Warning: {sc_cfg['frames_dir']} does not exist, skipping.")
            continue

        out_sub = base_out / sc_cfg["name"]
        composite_swimmer_into_video(
            frames_dir=sc_cfg["frames_dir"],
            output_dir=out_sub,
            swimmer_rgba=sc_cfg["swimmer_rgba"],
            start_pos=sc_cfg["start_pos"],
            end_pos=sc_cfg["end_pos"],
            scale=sc_cfg["scale"],
            base_angle=sc_cfg["base_angle"],
            submerged_axis=sc_cfg["submerged_axis"],
            depth_intensity=sc_cfg["depth_intensity"],
            scenario_name=sc_cfg["name"],
            title_label=sc_cfg["title"],
        )

    print("\n" + "=" * 60)
    print(f"All video sequences generated in '{base_out}'.")
    print("=" * 60)


if __name__ == "__main__":
    main()
