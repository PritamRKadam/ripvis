"""
compare_generation_methods.py
─────────────────────────────
Generates a high-resolution side-by-side visual comparison grid between:
1. Legacy Copy-Paste / Poisson Blended Swimmers (Method A)
2. Fine-Tuned Diffusion Generated / Inpainted Swimmers (Method B)
"""

import argparse
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont


def create_comparison(
    legacy_dir: Path,
    diffusion_dir: Path,
    output_path: Path,
    n_samples: int = 4,
    img_size: int = 448
):
    legacy_files = sorted(list(legacy_dir.glob("*.jpg")) + list(legacy_dir.glob("*.png")))
    diffusion_files = sorted(list(diffusion_dir.glob("*.jpg")) + list(diffusion_dir.glob("*.png")))

    if not legacy_files:
        for alt in [Path("data/synthetic_afo_people/images"), Path("data/Output1/images"), Path("data/synthetic/images"), Path("data/synthetic_afo_swimmers/images")]:
            if alt.exists():
                legacy_files = sorted(list(alt.glob("*.jpg")) + list(alt.glob("*.png")))
                if legacy_files:
                    break

    if not diffusion_files:
        print(f"No diffusion images found in {diffusion_dir}. Please run generation first.")
        return

    n_rows = min(n_samples, len(legacy_files), len(diffusion_files))
    if n_rows == 0:
        print("Not enough images to compare.")
        return

    header_h = 60
    card_pad = 16
    grid_w = img_size * 2 + card_pad * 3
    grid_h = header_h + n_rows * (img_size + card_pad) + card_pad

    # Modern dark slate background
    canvas = Image.new("RGB", (grid_w, grid_h), (22, 27, 34))
    draw = ImageDraw.Draw(canvas)

    # Header titles
    draw.rectangle([(card_pad, 12), (card_pad + img_size, 48)], fill=(45, 20, 24))
    draw.text((card_pad + 20, 22), "Method A: Legacy Cutout Blending (Heuristic)", fill=(240, 100, 100))

    draw.rectangle([(card_pad * 2 + img_size, 12), (card_pad * 2 + img_size * 2, 48)], fill=(18, 45, 28))
    draw.text((card_pad * 2 + img_size + 20, 22), "Method B: Fine-Tuned Diffusion Model (LoRA + DPM++)", fill=(80, 220, 140))

    for i in range(n_rows):
        y_top = header_h + i * (img_size + card_pad)

        # Legacy image
        leg_img = Image.open(legacy_files[i % len(legacy_files)]).convert("RGB")
        leg_img = leg_img.resize((img_size, img_size), Image.Resampling.LANCZOS)
        canvas.paste(leg_img, (card_pad, y_top))

        # Diffusion image
        diff_img = Image.open(diffusion_files[i % len(diffusion_files)]).convert("RGB")
        diff_img = diff_img.resize((img_size, img_size), Image.Resampling.LANCZOS)
        canvas.paste(diff_img, (card_pad * 2 + img_size, y_top))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path, "JPEG", quality=95)
    print(f"✓ Comparison visual saved to: {output_path}")


def parse_args():
    p = argparse.ArgumentParser(description="Compare legacy cut-and-paste vs diffusion generation.")
    p.add_argument("--legacy_dir", type=str, default="data/synthetic_afo_swimmers/images")
    p.add_argument("--diffusion_dir", type=str, default="data/synthetic_diffusion/images")
    p.add_argument("--output_path", type=str, default="runs/comparison_cutout_vs_diffusion.jpg")
    p.add_argument("--n_samples", type=int, default=4)
    p.add_argument("--img_size", type=int, default=448)
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    create_comparison(
        legacy_dir=Path(args.legacy_dir),
        diffusion_dir=Path(args.diffusion_dir),
        output_path=Path(args.output_path),
        n_samples=args.n_samples,
        img_size=args.img_size
    )
