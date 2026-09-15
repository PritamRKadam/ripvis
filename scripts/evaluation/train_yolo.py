"""
train_yolo.py
─────────────
Train a YOLOv8 instance segmentation model on the RipVIS synthetic dataset.

Wraps the Ultralytics YOLO API with RipVIS-specific defaults:
  - Model: yolov8n-seg (nano, fastest; or m-seg, l-seg for more capacity)
  - Classes: 1 (rip_current)
  - Augmentation: mosaic, mixup, HSV jitter, random flip
  - Metric: mAP@0.5 (mask IoU) + F2 score
  - GPU: auto-detected, falls back to CPU

Usage:
    # Quick smoke test (3 epochs)
    python3 scripts/train_yolo.py --quick

    # Standard training (50 epochs, GPU 0)
    python3 scripts/train_yolo.py \
        --data data/yolo_dataset/data.yaml \
        --model yolov8n-seg \
        --epochs 50 \
        --batch 16 \
        --device 0

    # Larger model (better accuracy, more VRAM)
    python3 scripts/train_yolo.py --model yolov8s-seg --epochs 100 --batch 8
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))


def parse_args():
    p = argparse.ArgumentParser(
        description="Train YOLOv8-seg on RipVIS synthetic rip current dataset."
    )
    p.add_argument("--data",      type=str,
                   default="data/yolo_dataset/data.yaml",
                   help="Path to data.yaml (default: data/yolo_dataset/data.yaml)")
    p.add_argument("--model",     type=str, default="yolov8n-seg",
                   help="YOLOv8 model variant or path to custom checkpoint (.pt) (default: yolov8n-seg)")
    p.add_argument("--epochs",    type=int, default=50,
                   help="Training epochs (default: 50)")
    p.add_argument("--batch",     type=int, default=16,
                   help="Batch size (default: 16; use 8 for l/x models)")
    p.add_argument("--imgsz",     type=int, default=640,
                   help="Input image size (default: 640)")
    p.add_argument("--device",    type=str, default="0",
                   help="Device: 0=GPU0, cpu (default: 0)")
    p.add_argument("--project",   type=str, default="runs/train",
                   help="Output directory base (default: runs/train)")
    p.add_argument("--name",      type=str, default="ripvis",
                   help="Experiment name (default: ripvis)")
    p.add_argument("--resume",    action="store_true",
                   help="Resume from last checkpoint")
    p.add_argument("--pretrained",action="store_true", default=True,
                   help="Use COCO pretrained weights (default: True)")
    p.add_argument("--lr0",       type=float, default=0.01,
                   help="Initial learning rate (default: 0.01)")
    p.add_argument("--patience",  type=int, default=20,
                   help="Early stopping patience (default: 20)")
    p.add_argument("--workers",   type=int, default=4,
                   help="DataLoader workers (default: 4)")
    p.add_argument("--cache",     action="store_true",
                   help="Cache images in RAM for faster training")
    p.add_argument("--exist_ok",  action="store_true",
                   help="Overwrite existing experiment dir")
    p.add_argument("--quick",     action="store_true",
                   help="Smoke test: 3 epochs, batch=4, no val early stop")
    p.add_argument("--generate",  action="store_true",
                   help="Auto-generate training data if data.yaml not found")
    return p.parse_args()


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

def check_ultralytics():
    """Verify ultralytics is installed."""
    try:
        import ultralytics
        print(f"  ultralytics {ultralytics.__version__} ✓")
        return True
    except ImportError:
        print("ERROR: ultralytics not installed.")
        print("  Install: pip3 install ultralytics")
        return False


def check_cuda():
    """Report GPU status."""
    try:
        import torch
        if torch.cuda.is_available():
            name = torch.cuda.get_device_name(0)
            mem  = torch.cuda.get_device_properties(0).total_memory // (1024**3)
            print(f"  GPU: {name} ({mem} GB) ✓")
            return True
        else:
            print("  GPU: not available, using CPU (training will be slow)")
            return False
    except Exception as e:
        print(f"  GPU check failed: {e}")
        return False


def maybe_generate_data(data_yaml: Path):
    """Auto-generate training data if it doesn't exist."""
    if data_yaml.exists():
        return

    print(f"  data.yaml not found at {data_yaml}. Generating synthetic data...")
    import subprocess
    result = subprocess.run(
        [sys.executable, "scripts/generate_training_data.py",
         "--n_images", "200", "--output", str(data_yaml.parent)],
        cwd=Path(__file__).parent.parent
    )
    if result.returncode != 0:
        raise RuntimeError("Failed to generate training data.")
    print("  Data generation complete.")


# ──────────────────────────────────────────────────────────────────────────────
# Training
# ──────────────────────────────────────────────────────────────────────────────

def build_augmentation_config(quick: bool = False) -> dict:
    """
    Return augmentation hyperparameters tuned for rip current detection.

    Rip currents are narrow water channels — we want:
      - Moderate scale/translation changes (rip currents can be anywhere)
      - No 90° rotation (beaches are always horizontal)
      - Color jitter (water varies in color)
      - Mosaic disabled for small datasets (causes artifacts)
    """
    if quick:
        return {
            "hsv_h": 0.015, "hsv_s": 0.7, "hsv_v": 0.4,
            "degrees": 5.0,   "translate": 0.1, "scale": 0.3,
            "shear": 0.0,     "perspective": 0.0,
            "flipud": 0.0,    "fliplr": 0.5,
            "mosaic": 0.5,    "mixup": 0.0,
            "copy_paste": 0.0
        }

    return {
        # Color augmentation (important for water appearance)
        "hsv_h": 0.015,    # Hue shift ±1.5% — subtle
        "hsv_s": 0.7,      # Saturation jitter — water turbidity varies
        "hsv_v": 0.4,      # Brightness jitter — lighting conditions

        # Geometric (conservative — beach orientation matters)
        "degrees": 5.0,    # Small rotation only
        "translate": 0.15, # Position shift
        "scale": 0.5,      # Scale change (rip currents vary in size)
        "shear": 2.0,      # Very mild shear

        # Flip
        "flipud": 0.0,     # Never flip vertically (sky must be up)
        "fliplr": 0.5,     # Horizontal flip (rip current can be left or right)

        # Mosaic/mixup
        "mosaic": 0.8,     # Mix 4 images — helps with small dataset
        "mixup": 0.1,      # Blend two images
        "copy_paste": 0.2  # Copy-paste augmentation (extra synthetic rips)
    }


def train(args):
    """Main training entry point."""
    from ultralytics import YOLO

    data_yaml = Path(args.data)
    project   = Path(args.project)

    # Check / generate data
    if not data_yaml.exists():
        if args.generate:
            maybe_generate_data(data_yaml)
        else:
            raise FileNotFoundError(
                f"data.yaml not found: {data_yaml}\n"
                "Run: python3 scripts/generate_training_data.py first,\n"
                "or use --generate flag."
            )

    # Load or download model
    model_spec = args.model
    if not args.pretrained:
        # Train from scratch (not recommended for small datasets)
        model_spec = args.model.replace(".pt", "") + ".yaml"

    print(f"  Loading model: {model_spec}")
    model = YOLO(model_spec)

    # Build training hyperparameters
    aug_cfg = build_augmentation_config(quick=args.quick)

    train_kwargs = {
        "data":      str(data_yaml),
        "epochs":    3 if args.quick else args.epochs,
        "batch":     4 if args.quick else args.batch,
        "imgsz":     args.imgsz,
        "device":    args.device,
        "project":   str(project),
        "name":      args.name,
        "exist_ok":  args.exist_ok or args.quick,
        "resume":    args.resume,
        "lr0":       args.lr0,
        "patience":  5 if args.quick else args.patience,
        "workers":   2 if args.quick else args.workers,
        "cache":     args.cache,
        "verbose":   True,
        "plots":     True,          # Save training curves
        "save":      True,          # Save best.pt + last.pt
        "save_period": 10,          # Save checkpoint every 10 epochs
        # Segmentation-specific
        "overlap_mask": True,       # Allow overlapping masks
        "mask_ratio": 4,            # Mask downsampling ratio
        # Augmentation
        **aug_cfg
    }

    print(f"\n  Training config:")
    print(f"    Model:   {model_spec}")
    print(f"    Data:    {data_yaml}")
    print(f"    Epochs:  {train_kwargs['epochs']}")
    print(f"    Batch:   {train_kwargs['batch']}")
    print(f"    ImgSz:   {args.imgsz}")
    print(f"    Device:  {args.device}")
    print()

    # Train
    results = model.train(**train_kwargs)

    # Report results
    print("\n" + "=" * 60)
    print("  Training complete!")
    exp_dir = project / args.name
    best_pt = exp_dir / "weights" / "best.pt"
    if best_pt.exists():
        print(f"  Best weights: {best_pt}")
    print(f"  Results dir:  {exp_dir}")

    return results, best_pt if best_pt.exists() else None


def run_val(model_path: Path, data_yaml: Path, device: str = "0"):
    """Run validation on the best checkpoint and print metrics."""
    from ultralytics import YOLO

    print(f"\nRunning validation: {model_path}")
    model   = YOLO(str(model_path))
    metrics = model.val(data=str(data_yaml), device=device, plots=True)

    box_map50  = metrics.box.map50   if hasattr(metrics, "box")  else None
    mask_map50 = metrics.seg.map50   if hasattr(metrics, "seg")  else None
    mask_map   = metrics.seg.map     if hasattr(metrics, "seg")  else None

    print("\nValidation Metrics:")
    if box_map50  is not None: print(f"  Box  mAP@0.5:     {box_map50:.4f}")
    if mask_map50 is not None: print(f"  Mask mAP@0.5:     {mask_map50:.4f}")
    if mask_map   is not None: print(f"  Mask mAP@0.5:0.95 {mask_map:.4f}")

    return metrics


# ──────────────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()

    print("=" * 60)
    print("  RipVIS — YOLOv8 Segmentation Training")
    print("=" * 60)

    if not check_ultralytics():
        sys.exit(1)

    has_gpu = check_cuda()
    if not has_gpu and args.device == "0":
        print("  Switching to CPU (no CUDA available)")
        args.device = "cpu"

    if args.quick:
        print("  [QUICK MODE] 3 epochs, batch=4")

    results, best_pt = train(args)

    # Auto-validate after training
    if best_pt and best_pt.exists():
        run_val(best_pt, Path(args.data), args.device)
        print(f"\n  Next step:")
        print(f"  python3 scripts/evaluate_model.py --checkpoint {best_pt}")


if __name__ == "__main__":
    main()
