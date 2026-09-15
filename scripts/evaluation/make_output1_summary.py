"""
make_output1_summary.py
─────────────────────────
Generates a visual 4x4 comparison grid of Output1 generated images
and symlinks Output1 to project root.
"""

import os
from pathlib import Path
import numpy as np
from PIL import Image
import cv2

def make_summary():
    output1_dir = Path("data/Output1")
    img_dir = output1_dir / "images"
    if not img_dir.exists():
        print(f"Directory '{img_dir}' does not exist.")
        return

    img_files = sorted(list(img_dir.glob("*.png")) + list(img_dir.glob("*.jpg")))[:16]
    if not img_files:
        print("No images found to summarize.")
        return

    # Create 4x4 grid
    cell_w, cell_h = 320, 320
    rows, cols = 4, 4
    grid = np.zeros((rows * cell_h, cols * cell_w, 3), dtype=np.uint8)

    for idx, p in enumerate(img_files):
        r = idx // cols
        c = idx % cols
        img = Image.open(p).convert("RGB")
        resized = np.array(img.resize((cell_w, cell_h), Image.Resampling.BILINEAR))
        grid[r*cell_h:(r+1)*cell_h, c*cell_w:(c+1)*cell_w] = resized

    grid_path = output1_dir / "preview_grid.png"
    Image.fromarray(grid).save(grid_path)
    print(f"Saved 4x4 summary grid to '{grid_path}'.")

    # Create symlink at project root for convenience
    root_link = Path("Output1")
    if not root_link.exists():
        try:
            root_link.symlink_to("data/Output1", target_is_directory=True)
            print("Created symlink 'Output1' -> 'data/Output1'.")
        except Exception as e:
            print(f"Symlink creation skipped: {e}")

if __name__ == "__main__":
    make_summary()
