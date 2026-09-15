#!/usr/bin/env python3
"""
scrape_v2_dataset.py
────────────────────
High-fidelity web scraping engine for the RipVIS V2 dataset.

Scrapes high-resolution aerial drone and coastal surveillance imagery:
1. Rip currents (seaward plumes, sandbar channels, surf conduits)
2. Swimmers in open ocean and surf zones
3. Coastal surf zones and breaking waves

Sources:
- Bing Image Crawler (via icrawler)
- Wikimedia Commons API (full resolution original files)

Outputs to data/V2/raw/ and logs metadata to data/V2/metadata/scrape_manifest.json.
"""

import argparse
import hashlib
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Set
from urllib.parse import urlparse

import requests
from PIL import Image
from tqdm import tqdm

try:
    from icrawler.builtin import BingImageCrawler
except ImportError:
    print("icrawler is required. Please install via: pip install icrawler")
    sys.exit(1)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger("RipVIS_V2_Scraper")


def compute_file_hash(filepath: Path) -> str:
    """Computes SHA-256 hash of a file."""
    hasher = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(65536):
            hasher.update(chunk)
    return hasher.hexdigest()


def scrape_wikimedia_commons(
    queries: List[str],
    output_dir: Path,
    max_images_per_query: int = 25,
    user_agent: str = "RipVISV2Research/2.0 (academic@cvl.ripvis)"
) -> List[Dict]:
    """
    Scrapes full-resolution bitmap images from Wikimedia Commons using the official API.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_records = []
    headers = {"User-Agent": user_agent}
    api_url = "https://commons.wikimedia.org/w/api.php"

    for query in queries:
        logger.info(f"[Wikimedia] Query: '{query}'...")
        params = {
            "action": "query",
            "generator": "search",
            "gsrsearch": f"{query} filetype:bitmap",
            "gsrnamespace": 6,  # File namespace
            "gsrlimit": max_images_per_query,
            "prop": "imageinfo",
            "iiprop": "url|size|mime|sha1",
            "format": "json"
        }

        try:
            resp = requests.get(api_url, params=params, headers=headers, timeout=15)
            if resp.status_code != 200:
                logger.warning(f"Wikimedia returned status {resp.status_code}")
                continue

            data = resp.json()
            pages = data.get("query", {}).get("pages", {})

            for page_id, page_info in pages.items():
                image_info_list = page_info.get("imageinfo", [])
                if not image_info_list:
                    continue

                info = image_info_list[0]
                img_url = info.get("url")
                mime = info.get("mime", "")
                width = info.get("width", 0)
                height = info.get("height", 0)

                # Skip non-images or low resolution (< 600px)
                if not mime.startswith("image/") or width < 600 or height < 600:
                    continue

                ext = Path(urlparse(img_url).path).suffix.lower()
                if ext not in [".jpg", ".jpeg", ".png"]:
                    ext = ".jpg"

                filename = f"wm_{page_id}{ext}"
                dest_path = output_dir / filename

                if dest_path.exists():
                    continue

                try:
                    img_resp = requests.get(img_url, headers=headers, timeout=20, stream=True)
                    if img_resp.status_code == 200:
                        with open(dest_path, "wb") as f:
                            for chunk in img_resp.iter_content(chunk_size=16384):
                                f.write(chunk)

                        # Verify image validity
                        with Image.open(dest_path) as im:
                            w, h = im.size

                        file_hash = compute_file_hash(dest_path)
                        record = {
                            "filename": filename,
                            "category": output_dir.name,
                            "source": "wikimedia_commons",
                            "query": query,
                            "url": img_url,
                            "width": w,
                            "height": h,
                            "sha256": file_hash,
                            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
                        }
                        manifest_records.append(record)
                        logger.info(f"  Saved {filename} ({w}x{h})")
                except Exception as e:
                    logger.debug(f"Failed to download {img_url}: {e}")
                    if dest_path.exists():
                        dest_path.unlink()

        except Exception as e:
            logger.warning(f"Failed query on Wikimedia: {e}")

    return manifest_records


def scrape_bing(
    queries: List[str],
    output_dir: Path,
    max_images_per_query: int = 40,
) -> List[Dict]:
    """
    Scrapes aerial drone and surveillance imagery using Bing Image Crawler.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_records = []

    for q_idx, query in enumerate(queries):
        logger.info(f"[Bing] Query: '{query}' (limit={max_images_per_query})...")
        temp_dir = output_dir / f"_temp_q{q_idx}"
        temp_dir.mkdir(parents=True, exist_ok=True)

        try:
            crawler = BingImageCrawler(
                storage={"root_dir": str(temp_dir)},
                log_level=logging.WARNING,
                downloader_threads=4
            )
            crawler.crawl(keyword=query, max_num=max_images_per_query)

            # Move and index downloaded images
            for p in temp_dir.glob("*.*"):
                if p.suffix.lower() not in [".jpg", ".jpeg", ".png", ".webp"]:
                    p.unlink()
                    continue

                try:
                    with Image.open(p) as im:
                        w, h = im.size
                        # Skip small icons/thumbnails
                        if w < 500 or h < 500:
                            p.unlink()
                            continue

                    file_hash = compute_file_hash(p)
                    new_filename = f"bing_{output_dir.name[:4]}_{q_idx}_{file_hash[:10]}{p.suffix.lower()}"
                    final_path = output_dir / new_filename

                    p.rename(final_path)

                    record = {
                        "filename": new_filename,
                        "category": output_dir.name,
                        "source": "bing_images",
                        "query": query,
                        "url": "bing_search",
                        "width": w,
                        "height": h,
                        "sha256": file_hash,
                        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
                    }
                    manifest_records.append(record)
                except Exception:
                    if p.exists():
                        p.unlink()

        except Exception as e:
            logger.warning(f"Error during Bing crawl for '{query}': {e}")
        finally:
            if temp_dir.exists():
                for rem in temp_dir.glob("*"):
                    rem.unlink()
                temp_dir.rmdir()

    return manifest_records


def main():
    parser = argparse.ArgumentParser(description="RipVIS V2 Image Scraping Pipeline")
    parser.add_argument("--base_dir", type=str, default="data/V2",
                        help="Base directory for V2 dataset")
    parser.add_argument("--max_per_query", type=int, default=30,
                        help="Maximum images per search query")
    parser.add_argument("--categories", nargs="+", default=["rip_currents", "swimmers_ocean", "surf_coastal"],
                        help="Categories to scrape")
    args = parser.parse_args()

    base_dir = Path(args.base_dir)
    raw_dir = base_dir / "raw"
    meta_dir = base_dir / "metadata"
    meta_dir.mkdir(parents=True, exist_ok=True)

    manifest_path = meta_dir / "scrape_manifest.json"

    # Curated domain-specific aerial drone queries (broadened for large-scale collection)
    category_queries = {
        "rip_currents": [
            "aerial view rip current drone ocean",
            "rip current beach overhead aerial",
            "aerial drone rip current channel sandbar",
            "coastal rip current aerial photography",
            "rip tide aerial view ocean beach",
            "rip current drone 4k overhead",
            "sandbar rip current channel aerial",
            "aerial drone beach rip neck plume",
            "dangerous rip current ocean aerial photo",
            "drone photo rip current australia coastline",
            "drone aerial rip current north carolina hawaii",
            "overhead ocean circulation rip current drone"
        ],
        "swimmers_ocean": [
            "aerial drone view swimmer open ocean",
            "top down drone view swimmers in water",
            "swimmers in surf zone aerial drone",
            "lifeguard rescue swimmer ocean aerial",
            "aerial view people swimming beach surf",
            "aerial view swimmers practicing with floats open ocean",
            "drone open water swimmer buoy ocean overhead",
            "open water swimmer aerial photography drone",
            "aerial drone triathletes swimming ocean",
            "drone view swimmer caught in ocean current",
            "overhead aerial view swimmer coastal waves",
            "drone photography people swimming in sea"
        ],
        "surf_coastal": [
            "aerial drone coastal surf zone waves",
            "overhead drone breaking waves beach coastline",
            "aerial view ocean waves surf zone sandbar",
            "drone top down waves breaking on sandbar",
            "coastal drone surf zone wave patterns",
            "overhead aerial beach sandbar waves turquoise",
            "aerial photo ocean beach surf line 4k",
            "drone photography coastal surf line breaking swells"
        ]
    }

    all_records = []
    if manifest_path.exists():
        try:
            with open(manifest_path, "r") as f:
                all_records = json.load(f)
        except Exception:
            all_records = []

    existing_hashes = {r["sha256"] for r in all_records if "sha256" in r}

    for cat in args.categories:
        if cat not in category_queries:
            continue

        queries = category_queries[cat]
        cat_dir = raw_dir / cat
        cat_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"\n==========================================")
        logger.info(f"  Scraping Category: {cat}")
        logger.info(f"==========================================")

        # 1. Scrape Wikimedia Commons
        wm_records = scrape_wikimedia_commons(
            queries=queries,
            output_dir=cat_dir,
            max_images_per_query=args.max_per_query
        )
        for r in wm_records:
            if r["sha256"] not in existing_hashes:
                all_records.append(r)
                existing_hashes.add(r["sha256"])

        # 2. Scrape Bing Images
        bing_records = scrape_bing(
            queries=queries,
            output_dir=cat_dir,
            max_images_per_query=args.max_per_query
        )
        for r in bing_records:
            if r["sha256"] not in existing_hashes:
                all_records.append(r)
                existing_hashes.add(r["sha256"])

    # Save updated manifest
    with open(manifest_path, "w") as f:
        json.dump(all_records, f, indent=2)

    logger.info(f"\n✅ Scraping complete! Total recorded entries: {len(all_records)}")
    logger.info(f"Manifest written to: {manifest_path}")


if __name__ == "__main__":
    main()
