#!/usr/bin/env python3
"""
audit_swimmer_videos.py
───────────────────────
Scans all extracted video frame sequences in data/V2/videos/frames/
using YOLOv8 person detection and HSV water-context analysis.

Strictly verifies that:
1. Human person(s) are detected (YOLO class 0).
2. The persons are actually in the water (ocean/sea/lake/river/surf foam).
3. Non-swimmers (kayaks without swimmers, stage shows, car rallies, empty waves) are pruned.
"""

import shutil
import json
from pathlib import Path
import cv2
import numpy as np
from ultralytics import YOLO

EXCLUDE_TITLES = [
    "mama_cax", "chromat", "fashion", "runway", "catwalk",
    "hurricane", "satellite", "radar", "typhoon",
    "kayak", "canoe", "rowing", "boat", "yacht", "ferry",
    "flume", "pilotversuch", "driftstr", "brandungsstr",
    "pajara", "avenida", "costco", "miley", "rigour", "schiff",
    "gar_fin", "fin_moore", "rallye", "border_pipe", "horse",
    "dolphin", "whale", "shark", "turtle", "jellyfish", "coral",
    "b-roll_video", "nature_reserve"
]


def is_title_excluded(name: str) -> bool:
    nl = name.lower()
    return any(k in nl for k in EXCLUDE_TITLES)


def compute_water_mask(img_bgr: np.ndarray) -> np.ndarray:
    """Computes a binary mask of water pixels in HSV space."""
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    # Blue / cyan / teal water
    mask_blue = (hsv[:, :, 0] >= 75) & (hsv[:, :, 0] <= 135) & (hsv[:, :, 1] >= 20) & (hsv[:, :, 2] >= 20)
    # Green / emerald / algae fresh or coastal water
    mask_green = (hsv[:, :, 0] >= 32) & (hsv[:, :, 0] < 75) & (hsv[:, :, 1] >= 25) & (hsv[:, :, 2] >= 25)
    # Breaking wave surf / whitewater / foam
    mask_foam = (hsv[:, :, 1] < 55) & (hsv[:, :, 2] > 160)
    # Deep / dark ocean water
    mask_dark_water = (hsv[:, :, 0] >= 70) & (hsv[:, :, 0] <= 140) & (hsv[:, :, 2] >= 15) & (hsv[:, :, 2] <= 70) & (hsv[:, :, 1] >= 25)

    return mask_blue | mask_green | mask_foam | mask_dark_water


def evaluate_frame(img_bgr: np.ndarray, model) -> dict:
    """Evaluates a single frame for person detections and water context."""
    h, w = img_bgr.shape[:2]
    water_mask = compute_water_mask(img_bgr)
    frame_water_ratio = float(np.mean(water_mask))

    results = model(img_bgr, verbose=False)
    persons = []
    for r in results:
        for b in r.boxes:
            if int(b.cls[0]) == 0 and float(b.conf[0]) >= 0.25:
                persons.append(b.xyxy[0].cpu().numpy())

    person_water_contexts = []
    for box in persons:
        x1, y1, x2, y2 = [int(v) for v in box]
        bw = max(1, x2 - x1)
        bh = max(1, y2 - y1)
        # Expand box by 35% to examine surrounding water envelope
        ex1 = max(0, x1 - int(bw * 0.35))
        ey1 = max(0, y1 - int(bh * 0.35))
        ex2 = min(w, x2 + int(bw * 0.35))
        ey2 = min(h, y2 + int(bh * 0.35))

        ctx = float(np.mean(water_mask[ey1:ey2, ex1:ex2]))
        person_water_contexts.append(ctx)

    avg_pw = float(np.mean(person_water_contexts)) if person_water_contexts else 0.0
    return {
        "num_persons": len(persons),
        "frame_water_ratio": frame_water_ratio,
        "avg_person_water_context": avg_pw,
        "persons_in_water": sum(1 for c in person_water_contexts if c >= 0.20)
    }


def main():
    model = YOLO("yolov8n.pt")

    base_dir = Path("data/V2/videos")
    frames_dir = base_dir / "frames"
    raw_dir = base_dir / "raw"
    previews_dir = base_dir / "previews"
    manifest_path = base_dir / "metadata" / "video_manifest.json"

    seq_dirs = sorted([d for d in frames_dir.iterdir() if d.is_dir()])
    print(f"Auditing {len(seq_dirs)} video sequences strictly for swimmers in water...")

    valid_swimmers = []
    rejected = []

    for sdir in seq_dirs:
        name = sdir.name
        if is_title_excluded(name):
            rejected.append((name, f"Title exclusion pattern matched"))
            print(f"  [REJECT - TITLE] {name}")
            continue

        frame_files = sorted(list(sdir.glob("*.jpg")))
        if not frame_files:
            rejected.append((name, "No frames"))
            continue

        step = max(1, len(frame_files) // 10)
        sample_frames = frame_files[::step][:10]

        frames_with_swimmer = 0
        total_persons = 0
        frame_water_ratios = []
        person_water_contexts = []

        for fpath in sample_frames:
            img = cv2.imread(str(fpath))
            if img is None:
                continue
            ev = evaluate_frame(img, model)
            frame_water_ratios.append(ev["frame_water_ratio"])
            if ev["num_persons"] > 0:
                total_persons += ev["num_persons"]
                person_water_contexts.append(ev["avg_person_water_context"])
                if ev["persons_in_water"] > 0 or (ev["avg_person_water_context"] >= 0.20 and ev["frame_water_ratio"] >= 0.20):
                    frames_with_swimmer += 1

        mean_fw = float(np.mean(frame_water_ratios)) if frame_water_ratios else 0.0
        mean_pw = float(np.mean(person_water_contexts)) if person_water_contexts else 0.0

        # Swimmer video criteria:
        # 1. At least 2 sampled frames have a person confirmed in water context
        # 2. Or >= 20% of frames have persons in water with substantial water in scene
        is_swimmer = (frames_with_swimmer >= 2 and mean_fw >= 0.20) or (frames_with_swimmer >= 1 and mean_pw >= 0.40 and mean_fw >= 0.35)

        if is_swimmer:
            valid_swimmers.append({
                "name": name,
                "swimmer_frames": frames_with_swimmer,
                "samples": len(sample_frames),
                "mean_frame_water": round(mean_fw, 2),
                "mean_person_water": round(mean_pw, 2)
            })
            print(f"  ✓ [SWIMMER IN WATER] {name} (frames: {frames_with_swimmer}/{len(sample_frames)}, water: {mean_fw:.2f}, person_ctx: {mean_pw:.2f})")
        else:
            reason = f"persons={total_persons}, swimmer_frames={frames_with_swimmer}/{len(sample_frames)}, mean_fw={mean_fw:.2f}, mean_pw={mean_pw:.2f}"
            rejected.append((name, reason))
            print(f"  ✗ [REJECT] {name} ({reason})")

    print("\n" + "=" * 60)
    print(f"Audit Summary:")
    print(f"  Confirmed Swimmers in Water : {len(valid_swimmers)}")
    print(f"  Rejected Non-Swimmers       : {len(rejected)}")
    print("=" * 60)

    # Clean rejected sequences
    for name, reason in rejected:
        fdir = frames_dir / name
        if fdir.exists():
            shutil.rmtree(fdir)

        for rf in raw_dir.rglob("*.*"):
            if rf.stem == name:
                rf.unlink()
                print(f"  Pruned raw file: {rf.name}")

        for p in previews_dir.glob(f"{name}*"):
            p.unlink()

    # Update manifest
    if manifest_path.exists():
        try:
            with open(manifest_path, "r") as f:
                records = json.load(f)
            valid_names = {v["name"] for v in valid_swimmers}
            cleaned_records = [r for r in records if Path(r.get("filename", "")).stem in valid_names]
            with open(manifest_path, "w") as f:
                json.dump(cleaned_records, f, indent=2)
            print(f"Updated manifest: {len(cleaned_records)} entries remaining.")
        except Exception as e:
            print(f"Error updating manifest: {e}")

    print(f"\nTotal verified swimmer video sequences remaining: {len(valid_swimmers)}")


if __name__ == "__main__":
    main()
