#!/usr/bin/env python3
"""
scrape_v2_videos.py
───────────────────
High-fidelity video scraper targeting strictly HUMAN SWIMMERS IN WATER
(open ocean, coastal surf zone, lakes, rivers, and wild swimming).

Key Features:
- In-line Swimmer & Water Verification using YOLOv8 person detection and HSV water-context analysis.
  Any clip lacking confirmed people immersed in water is deleted immediately upon download.
- Automatic 4x4 preview contact sheet and animated WebP generation for instant inspection.
- Dual scraping strategy: Wikimedia Commons Search queries + Category Crawling (e.g. Category:Open water swimming).
- Strict exclude list against biology papers, animals, fashion shows, kayaks, satellites, shipwrecks, and paddleboards.
- Manifest maintained at data/V2/videos/metadata/video_manifest.json.
"""

import argparse
import hashlib
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple
from urllib.parse import urlparse

import cv2
import numpy as np
import requests
from PIL import Image
from ultralytics import YOLO

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger("RipVIS_SwimmerScraper")

EXCLUDE_KEYWORDS = [
    "pone.", "s00", "journal.pone", "plos", "microscopy", "bacteria",
    "cilia", "sperm", "embryo", "cells", "in-vitro", "in-vivo",
    "zebrafish", "fish", "larvae", "tadpole", "rodent", "rat", "mouse",
    "dolphin", "whale", "turtle", "shark", "drosophila", "daphnia",
    "otter", "seal", "penguin", "duck", "swan", "bird", "crab", "jellyfish",
    "coral", "nematode", "worm", "loon", "muskrat", "frog",
    "mama_cax", "chromat", "fashion", "runway", "catwalk", "model",
    "hurricane", "satellite", "radar", "typhoon", "ernesto",
    "kayak", "canoe", "rowing", "boat", "yacht", "ferry", "vessel", "ship",
    "cargo", "rigour", "transit", "schiff", "bridge", "swing_bridge",
    "flume", "pilotversuch", "orbitalbewegung", "driftstr", "brandungsstr",
    "gar_fin", "fin_moore", "rallye", "border_pipe", "horse", "diving_horse",
    "b-roll_video", "nature_reserve", "wreck", "shipwreck", "paddle",
    "paddleboard", "stand-up", "paddling", "aquarium", "eend", "alk", "vogel", "duiker"
]


def is_excluded(text: str) -> bool:
    t = text.lower()
    return any(k in t for k in EXCLUDE_KEYWORDS)


def compute_file_hash(filepath: Path) -> str:
    """Computes SHA-256 hash of a file."""
    hasher = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(65536):
            hasher.update(chunk)
    return hasher.hexdigest()


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


def generate_preview_artifacts(filepath: Path, previews_dir: Path, num_samples: int = 16):
    """Generates 4x4 contact sheet keyframe grid and animated WebP for inspection."""
    try:
        previews_dir.mkdir(parents=True, exist_ok=True)
        cap = cv2.VideoCapture(str(filepath))
        if not cap.isOpened():
            return

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total_frames < 8:
            cap.release()
            return

        indices = np.linspace(0, total_frames - 1, min(num_samples, total_frames), dtype=int)
        frames = []
        for idx in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
            ret, frame = cap.read()
            if ret and frame is not None:
                frames.append(frame)
        cap.release()

        if not frames:
            return

        # 4x4 grid
        rows, cols = 4, 4
        cell_w, cell_h = 320, 180
        grid = np.zeros((rows * cell_h, cols * cell_w, 3), dtype=np.uint8)
        for i, f in enumerate(frames[:16]):
            r = i // cols
            c = i % cols
            res = cv2.resize(f, (cell_w, cell_h))
            grid[r * cell_h:(r + 1) * cell_h, c * cell_w:(c + 1) * cell_w] = res

        grid_path = previews_dir / f"{filepath.stem}_keyframes.jpg"
        cv2.imwrite(str(grid_path), grid, [int(cv2.IMWRITE_JPEG_QUALITY), 85])

        # Animated WebP
        webp_path = previews_dir / f"{filepath.stem}_preview.webp"
        pil_frames = [Image.fromarray(cv2.cvtColor(cv2.resize(f, (480, 270)), cv2.COLOR_BGR2RGB)) for f in frames[:16]]
        if pil_frames:
            pil_frames[0].save(
                webp_path,
                save_all=True,
                append_images=pil_frames[1:],
                duration=350,
                loop=0,
                quality=80
            )
    except Exception as e:
        logger.debug(f"Preview generation error for {filepath.name}: {e}")


def verify_swimmer_in_water(filepath: Path, model, num_samples: int = 12) -> Tuple[bool, Dict]:
    """
    Decodes sampled frames from a video file and validates:
    1. Human person(s) (YOLO class 0) are present in the video.
    2. Persons are in an aquatic/water environment (frame water + surrounding water context).
    """
    try:
        cap = cv2.VideoCapture(str(filepath))
        if not cap.isOpened():
            return False, {}

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        duration_sec = total_frames / fps if fps > 0 else 0.0

        if total_frames < 8 or w < 320 or h < 240:
            cap.release()
            return False, {}

        sample_indices = np.linspace(0, total_frames - 1, min(num_samples, total_frames), dtype=int)
        
        frames_with_swimmer = 0
        total_person_detections = 0
        frame_water_ratios = []
        person_water_contexts = []

        for idx in sample_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
            ret, frame = cap.read()
            if not ret or frame is None:
                continue

            water_mask = compute_water_mask(frame)
            fw_ratio = float(np.mean(water_mask))
            frame_water_ratios.append(fw_ratio)

            # Run YOLO
            results = model(frame, verbose=False)
            frame_persons = []
            for r in results:
                for b in r.boxes:
                    if int(b.cls[0]) == 0 and float(b.conf[0]) >= 0.25:
                        frame_persons.append(b.xyxy[0].cpu().numpy())

            if frame_persons:
                total_person_detections += len(frame_persons)
                pw_contexts = []
                for box in frame_persons:
                    x1, y1, x2, y2 = [int(v) for v in box]
                    bw = max(1, x2 - x1)
                    bh = max(1, y2 - y1)
                    ex1 = max(0, x1 - int(bw * 0.35))
                    ey1 = max(0, y1 - int(bh * 0.35))
                    ex2 = min(w, x2 + int(bw * 0.35))
                    ey2 = min(h, y2 + int(bh * 0.35))
                    ctx = float(np.mean(water_mask[ey1:ey2, ex1:ex2]))
                    pw_contexts.append(ctx)

                person_water_contexts.extend(pw_contexts)
                # Stricter swimmer rule: person must be surrounded by water
                if any(c >= 0.22 for c in pw_contexts) or (fw_ratio >= 0.35 and any(c >= 0.15 for c in pw_contexts)):
                    frames_with_swimmer += 1

        cap.release()

        mean_fw = float(np.mean(frame_water_ratios)) if frame_water_ratios else 0.0
        mean_pw = float(np.mean(person_water_contexts)) if person_water_contexts else 0.0

        # Acceptance rule:
        # At least 2 sampled frames with swimmer in water, scene has >= 20% water
        # OR at least 1 frame with swimmer in high-water scene (mean_pw >= 0.35 and mean_fw >= 0.35)
        is_valid = (frames_with_swimmer >= 2 and mean_fw >= 0.20) or (frames_with_swimmer >= 1 and mean_pw >= 0.35 and mean_fw >= 0.35)

        stats = {
            "width": w,
            "height": h,
            "fps": round(fps, 2),
            "total_frames": total_frames,
            "duration_sec": round(duration_sec, 2),
            "swimmer_frames": frames_with_swimmer,
            "total_persons": total_person_detections,
            "mean_frame_water": round(mean_fw, 2),
            "mean_person_water": round(mean_pw, 2)
        }
        return is_valid, stats
    except Exception as e:
        logger.debug(f"Video verification exception for {filepath.name}: {e}")
        return False, {}


def process_candidate_video(
    title: str,
    output_dir: Path,
    previews_dir: Path,
    existing_hashes: Set[str],
    existing_names: Set[str],
    model: YOLO,
    headers: Dict,
    query_or_category: str,
    max_size_mb: float,
    min_size_mb: float,
    api_url: str = "https://commons.wikimedia.org/w/api.php"
) -> Optional[Dict]:
    """Downloads candidate video from Wikimedia and verifies swimmer presence."""
    if is_excluded(title):
        return None

    p2 = {
        "action": "query",
        "titles": title,
        "prop": "imageinfo",
        "iiprop": "url|size|mime|sha1",
        "format": "json"
    }
    try:
        r2 = requests.get(api_url, params=p2, headers=headers, timeout=12)
        pages = r2.json().get("query", {}).get("pages", {})
    except Exception:
        return None

    for pid, p in pages.items():
        image_info = p.get("imageinfo", [])
        if not image_info:
            continue
        ii = image_info[0]
        video_url = ii.get("url", "")
        mime = ii.get("mime", "")
        width = ii.get("width", 0)
        height = ii.get("height", 0)
        size_bytes = ii.get("size", 0)
        size_mb = size_bytes / (1024 * 1024)

        if not video_url or is_excluded(video_url):
            continue

        if not mime.startswith("video/") and not mime.endswith("/ogg"):
            continue
        if width < 320 or height < 240:
            continue
        if size_mb > max_size_mb or size_mb < min_size_mb:
            continue

        url_ext = Path(urlparse(video_url).path).suffix.lower()
        if not url_ext:
            url_ext = ".webm"

        clean_name = title.replace("File:", "").replace(" ", "_").replace("/", "_")
        clean_name = "".join(c for c in clean_name if c.isalnum() or c in "._-")
        for common_ext in [".ogv", ".webm", ".mp4", ".mov", ".avi"]:
            if clean_name.lower().endswith(common_ext):
                clean_name = clean_name[:-len(common_ext)]
                break
        clean_name += url_ext

        if is_excluded(clean_name):
            continue

        if clean_name in existing_names:
            continue

        dest_path = output_dir / clean_name
        if dest_path.exists():
            existing_names.add(clean_name)
            continue

        logger.info(f"  Downloading candidate ({size_mb:.1f} MB): {clean_name}...")
        try:
            v_resp = requests.get(video_url, headers=headers, stream=True, timeout=50)
            if v_resp.status_code == 200:
                with open(dest_path, "wb") as f:
                    for chunk in v_resp.iter_content(chunk_size=65536):
                        f.write(chunk)

                # Strictly verify swimmer in water with YOLO and HSV water detector
                is_swimmer, stats = verify_swimmer_in_water(dest_path, model)
                if not is_swimmer:
                    logger.info(f"    ✗ Discarded: No verified swimmers in water ({dest_path.name}) - stats: {stats}")
                    if dest_path.exists():
                        dest_path.unlink()
                    continue

                file_hash = compute_file_hash(dest_path)
                if file_hash in existing_hashes:
                    logger.info(f"    ✗ Duplicate hash, skipping.")
                    if dest_path.exists():
                        dest_path.unlink()
                    continue

                existing_hashes.add(file_hash)
                existing_names.add(clean_name)

                # Generate previews immediately
                generate_preview_artifacts(dest_path, previews_dir)

                record = {
                    "video_id": dest_path.stem,
                    "filename": dest_path.name,
                    "category": output_dir.name,
                    "query": query_or_category,
                    "url": video_url,
                    "sha256": file_hash,
                    "size_mb": round(size_mb, 2),
                    "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                    **stats
                }
                logger.info(f"    ✅ CONFIRMED SWIMMER: {dest_path.name} ({stats['width']}x{stats['height']}, {stats['duration_sec']}s, {stats['swimmer_frames']} swimmer frames)")
                return record
        except Exception as e:
            logger.debug(f"Download exception for {clean_name}: {e}")
            if dest_path.exists():
                dest_path.unlink()

    return None


def search_and_download_wikimedia_videos(
    queries: List[str],
    categories: List[str],
    output_dir: Path,
    previews_dir: Path,
    existing_hashes: Set[str],
    existing_names: Set[str],
    model: YOLO,
    max_videos_per_query: int = 6,
    max_size_mb: float = 120.0,
    min_size_mb: float = 0.2,
    target_total: int = 50,
    current_count_func = None,
    user_agent: str = "RipVISResearch/2.0 (academic@cvl.ripvis)"
) -> List[Dict]:
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_records = []
    headers = {"User-Agent": user_agent}
    api_url = "https://commons.wikimedia.org/w/api.php"

    # 1. First search via Categories (curated collections of files)
    for cat in categories:
        if current_count_func and current_count_func() >= target_total:
            logger.info(f"Target total of {target_total} reached. Stopping.")
            break

        logger.info(f"\n[Wikimedia Category] Crawling '{cat}'...")
        cat_params = {
            "action": "query",
            "list": "categorymembers",
            "cmtitle": cat,
            "cmtype": "file",
            "cmlimit": 80,
            "format": "json"
        }
        try:
            r = requests.get(api_url, params=cat_params, headers=headers, timeout=14)
            members = r.json().get("query", {}).get("categorymembers", [])
            for m in members:
                if current_count_func and current_count_func() >= target_total:
                    break
                rec = process_candidate_video(
                    title=m["title"],
                    output_dir=output_dir,
                    previews_dir=previews_dir,
                    existing_hashes=existing_hashes,
                    existing_names=existing_names,
                    model=model,
                    headers=headers,
                    query_or_category=cat,
                    max_size_mb=max_size_mb,
                    min_size_mb=min_size_mb,
                    api_url=api_url
                )
                if rec:
                    manifest_records.append(rec)
        except Exception as e:
            logger.warning(f"Error querying category '{cat}': {e}")

    # 2. Next search via text queries
    for query in queries:
        if current_count_func and current_count_func() >= target_total:
            logger.info(f"Target total of {target_total} videos reached. Stopping scraper.")
            break

        logger.info(f"\n[Wikimedia Video] Query: '{query}'...")
        params = {
            "action": "query",
            "list": "search",
            "srsearch": f"{query} filetype:video",
            "srnamespace": 6,
            "srlimit": max_videos_per_query * 8,
            "format": "json"
        }

        try:
            resp = requests.get(api_url, params=params, headers=headers, timeout=14)
            if resp.status_code != 200:
                continue

            items = resp.json().get("query", {}).get("search", [])
            downloaded_for_query = 0

            for it in items:
                if current_count_func and current_count_func() >= target_total:
                    break
                if downloaded_for_query >= max_videos_per_query:
                    break

                rec = process_candidate_video(
                    title=it["title"],
                    output_dir=output_dir,
                    previews_dir=previews_dir,
                    existing_hashes=existing_hashes,
                    existing_names=existing_names,
                    model=model,
                    headers=headers,
                    query_or_category=query,
                    max_size_mb=max_size_mb,
                    min_size_mb=min_size_mb,
                    api_url=api_url
                )
                if rec:
                    manifest_records.append(rec)
                    downloaded_for_query += 1

        except Exception as e:
            logger.warning(f"Error querying Wikimedia videos for '{query}': {e}")

    return manifest_records


def main():
    parser = argparse.ArgumentParser(description="RipVIS V2 Strict Swimmer Video Scraper")
    parser.add_argument("--base_dir", type=str, default="data/V2/videos",
                        help="Base directory for V2 video dataset")
    parser.add_argument("--category", type=str, default="all",
                        choices=["all", "swimmers_open_water", "swimmers_surf_zone"],
                        help="Specific swimmer category to scrape")
    parser.add_argument("--add_count", type=int, default=None,
                        help="Number of new verified videos to add")
    parser.add_argument("--target_total", type=int, default=50,
                        help="Target total number of swimmer videos to collect")
    parser.add_argument("--max_per_query", type=int, default=8,
                        help="Max videos per individual query")
    parser.add_argument("--max_size_mb", type=float, default=120.0,
                        help="Maximum file size in MB")
    parser.add_argument("--min_size_mb", type=float, default=0.2,
                        help="Minimum file size in MB")
    args = parser.parse_args()

    base_dir = Path(args.base_dir)
    raw_dir = base_dir / "raw"
    previews_dir = base_dir / "previews"
    meta_dir = base_dir / "metadata"
    meta_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = meta_dir / "video_manifest.json"

    logger.info("Loading YOLOv8n detector for in-line swimmer validation...")
    model = YOLO("yolov8n.pt")

    # Curated Wikimedia categories containing direct swimming video uploads
    category_sources = {
        "swimmers_open_water": [
            "Category:Swimming in the sea",
            "Category:Swimmers in the sea",
            "Category:Open water swimming",
            "Category:Channel swimming",
            "Category:Marathon swimming",
            "Category:Wild swimming",
            "Category:Swimmers in water",
            "Category:Swimming in the ocean"
        ],
        "swimmers_surf_zone": [
            "Category:People swimming",
            "Category:Surf lifesaving",
            "Category:Water rescue",
            "Category:Beach bathers",
            "Category:Bathing in the sea"
        ]
    }

    # Rich multi-lingual search queries strictly for open ocean / sea swimmers
    category_queries = {
        "swimmers_open_water": [
            "open ocean swimming",
            "swimming in open ocean",
            "sea swimming open water",
            "swimmers in open sea",
            "open water ocean swimmer",
            "swimming in the sea water",
            "swimmers in deep sea water",
            "ocean swim race",
            "ocean crossing swimmer",
            "channel swimming ocean",
            "triathlon ocean swimming",
            "marathon swimming ocean",
            "swimmer waves open sea",
            "swimming in Pacific ocean",
            "swimming in Atlantic ocean",
            "swimming in Mediterranean sea",
            "freestyle swimmer in ocean",
            "wild sea swimming",
            "natation en pleine mer",
            "nageur eau libre ocean",
            "baignade en pleine mer",
            "Freiwasserschwimmen offenes Meer",
            "Schwimmer im Meer",
            "nadador en mar abierto",
            "nadando en el oceano",
            "nuotatore in mare aperto",
            "nuoto in acque libere mare",
            "nadadores mar abierto",
            "zwemmen in open zee"
        ],
        "swimmers_surf_zone": [
            "swimmers in water",
            "people swimming in ocean",
            "swimmers in surf waves",
            "bathers swimming sea",
            "beach swim water waves",
            "water rescue swimmer",
            "lifeguard swimmer rescue",
            "nadadores en el mar",
            "nageurs dans l'eau mer",
            "nuotatore in mare onde",
            "pessoas nadando praia",
            "Schwimmer im Wasser Meer",
            "zwemmen in zee golven",
            "bathers in water ocean",
            "bodyboarding in waves water"
        ]
    }

    all_records = []
    if manifest_path.exists():
        try:
            with open(manifest_path, "r") as f:
                all_records = json.load(f)
        except Exception:
            all_records = []

    def get_current_disk_count() -> int:
        return len(list(raw_dir.rglob("*.webm")) + list(raw_dir.rglob("*.mp4")) + list(raw_dir.rglob("*.ogv")))

    # Synchronize records with disk
    disk_names = {f.name for f in (list(raw_dir.rglob("*.webm")) + list(raw_dir.rglob("*.mp4")) + list(raw_dir.rglob("*.ogv")))}
    all_records = [r for r in all_records if r.get("filename") in disk_names]

    existing_hashes = {r["sha256"] for r in all_records if "sha256" in r}
    existing_names = {r["filename"] for r in all_records if "filename" in r}

    current_count = get_current_disk_count()
    if args.add_count:
        target_total = current_count + args.add_count
    else:
        target_total = args.target_total

    logger.info(f"Verified swimmer videos currently on disk: {current_count}. Target total: {target_total}")

    selected_categories = [args.category] if args.category != "all" else list(category_queries.keys())

    for cat_name in selected_categories:
        if get_current_disk_count() >= target_total:
            break

        queries = category_queries.get(cat_name, [])
        cat_dir = raw_dir / cat_name
        cat_members = category_sources.get(cat_name, [])

        logger.info(f"\n==========================================")
        logger.info(f"  Scraping Swimmer Category: {cat_name}")
        logger.info(f"==========================================")

        records = search_and_download_wikimedia_videos(
            queries=queries,
            categories=cat_members,
            output_dir=cat_dir,
            previews_dir=previews_dir,
            existing_hashes=existing_hashes,
            existing_names=existing_names,
            model=model,
            max_videos_per_query=args.max_per_query,
            max_size_mb=args.max_size_mb,
            min_size_mb=args.min_size_mb,
            target_total=target_total,
            current_count_func=get_current_disk_count
        )
        all_records.extend(records)
        logger.info(f"Category {cat_name} finished. Total valid videos on disk now: {get_current_disk_count()}")

        # Incremental save of manifest after each category
        with open(manifest_path, "w") as f:
            json.dump(all_records, f, indent=2)

    logger.info(f"\n" + "=" * 60)
    logger.info(f"  ✅ Video Scraping Complete! Total confirmed swimmer videos: {len(all_records)}")
    logger.info(f"  Total valid video files on disk: {get_current_disk_count()}")
    logger.info(f"  Manifest written to: {manifest_path}")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
