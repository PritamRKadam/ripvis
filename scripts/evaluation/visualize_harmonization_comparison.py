"""
visualize_harmonization_comparison.py
────────────────────────────────────────
Generates side-by-side visual comparison between:
1. Baseline (Raw alpha paste)
2. Deep Matting + LAB Relighting Harmonization
"""

import sys
from pathlib import Path
import numpy as np
from PIL import Image
import cv2

sys.path.insert(0, str(Path(__file__).parent.parent))
from utils.person_utils import place_person_cutout, load_person_cutouts

def generate_comparison():
    input_img_dir = Path("data/synthetic/imprinted/images")
    cutouts = load_person_cutouts("data/person_cutouts")
    if not cutouts:
        print("No cutouts found!")
        return

    img_files = sorted(list(input_img_dir.glob("*.png")) + list(input_img_dir.glob("*.jpg")))[:2]
    out_dir = Path("data/test_harmonized/comparison")
    out_dir.mkdir(parents=True, exist_ok=True)

    for idx, img_path in enumerate(img_files):
        img_np = np.array(Image.open(img_path).convert("RGB"))
        h, w = img_np.shape[:2]

        cutout = cutouts[idx % len(cutouts)]
        cx, cy = int(w * 0.5), int(h * 0.55)
        target_h = int(h * 0.18)

        # 1. Baseline
        base_img, _, _ = place_person_cutout(
            background_img=img_np.copy(),
            cutout_rgba=cutout,
            center_pos=(cx, cy),
            target_height_px=target_h,
            enable_harmonization=False,
            enable_deep_matting=False
        )

        # 2. Harmonized + Deep Matting
        harm_img, _, _ = place_person_cutout(
            background_img=img_np.copy(),
            cutout_rgba=cutout,
            center_pos=(cx, cy),
            target_height_px=target_h,
            enable_harmonization=True,
            enable_deep_matting=True,
            harmonization_strength=0.8
        )

        # Side-by-side canvas
        grid = np.zeros((h, w * 2, 3), dtype=np.uint8)
        grid[:, :w] = base_img
        grid[:, w:] = harm_img

        # Add text labels
        cv2.putText(grid, "Baseline (Raw Paste)", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(grid, "Deep Matting + LAB Relighting", (w + 20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 120), 2, cv2.LINE_AA)

        out_path = out_dir / f"comparison_{idx:02d}.png"
        Image.fromarray(grid).save(out_path)
        print(f"Saved comparison to '{out_path}'.")

if __name__ == "__main__":
    generate_comparison()
