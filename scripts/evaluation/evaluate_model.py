"""
evaluate_model.py
─────────────────
Evaluate a trained YOLOv8-seg checkpoint on RipVIS data.

Computes:
  - mAP@0.5 and mAP@0.5:0.95 (mask IoU)
  - F2 score (recall-weighted, matching RipVIS benchmark)
  - Per-image precision / recall
  - Saves prediction overlay images

Usage:
    python3 scripts/evaluate_model.py \
        --checkpoint runs/train/ripvis/weights/best.pt \
        --data data/yolo_dataset/data.yaml \
        --split val \
        --save_vis

    # On a custom image directory
    python3 scripts/evaluate_model.py \
        --checkpoint runs/train/ripvis/weights/best.pt \
        --images data/samples/images/ \
        --conf 0.25 --iou 0.5
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
from PIL import Image, ImageDraw, ImageFont


def parse_args():
    p = argparse.ArgumentParser(
        description="Evaluate YOLOv8-seg checkpoint on RipVIS rip current data."
    )
    p.add_argument("--checkpoint",  type=str, required=True,
                   help="Path to trained .pt checkpoint.")
    p.add_argument("--data",        type=str,
                   default="data/yolo_dataset/data.yaml",
                   help="data.yaml path (for official val set metrics).")
    p.add_argument("--images",      type=str, default=None,
                   help="Custom image directory for inference-only evaluation.")
    p.add_argument("--masks",       type=str, default=None,
                   help="Ground truth mask directory (for IoU computation).")
    p.add_argument("--split",       type=str, default="val",
                   choices=["train", "val"],
                   help="Dataset split to evaluate (default: val).")
    p.add_argument("--conf",        type=float, default=0.25,
                   help="Confidence threshold (default: 0.25).")
    p.add_argument("--iou",         type=float, default=0.45,
                   help="IoU NMS threshold (default: 0.45).")
    p.add_argument("--device",      type=str, default="0",
                   help="Device: 0=GPU0, cpu.")
    p.add_argument("--output",      type=str, default="runs/eval",
                   help="Output directory for visualizations.")
    p.add_argument("--save_vis",    action="store_true", default=True,
                   help="Save prediction overlay images.")
    p.add_argument("--n_vis",       type=int, default=16,
                   help="Number of visualization images to save (default: 16).")
    p.add_argument("--beta",        type=float, default=2.0,
                   help="Beta for F-beta score (default: 2.0 = F2, recall-weighted).")
    return p.parse_args()


# ──────────────────────────────────────────────────────────────────────────────
# F-beta score (RipVIS uses F2 to penalize missed rip currents heavily)
# ──────────────────────────────────────────────────────────────────────────────

def f_beta_score(precision: float, recall: float, beta: float = 2.0) -> float:
    """
    Compute F-beta score.

    F2 (beta=2) weights recall twice as much as precision.
    This is the primary RipVIS benchmark metric — missing a rip current
    (false negative) is much worse than a false alarm (false positive).
    """
    if precision + recall < 1e-9:
        return 0.0
    b2 = beta ** 2
    return (1 + b2) * precision * recall / (b2 * precision + recall)


def mask_iou(pred_mask: np.ndarray, gt_mask: np.ndarray) -> float:
    """Compute IoU between two binary masks."""
    pred_b = pred_mask > 0
    gt_b   = gt_mask   > 0
    inter  = (pred_b & gt_b).sum()
    union  = (pred_b | gt_b).sum()
    return float(inter / union) if union > 0 else 0.0


# ──────────────────────────────────────────────────────────────────────────────
# Official validation (YOLO metrics)
# ──────────────────────────────────────────────────────────────────────────────

def run_official_val(checkpoint: str, data_yaml: str, device: str,
                     conf: float, iou: float, output_dir: Path) -> dict:
    """Run YOLO's built-in val() and return metric dict."""
    from ultralytics import YOLO

    model   = YOLO(checkpoint)
    metrics = model.val(
        data=data_yaml,
        device=device,
        conf=conf,
        iou=iou,
        plots=True,
        project=str(output_dir),
        name="official_val",
        exist_ok=True,
        verbose=True
    )

    results = {}
    if hasattr(metrics, "box"):
        results["box_map50"]     = float(metrics.box.map50)
        results["box_map"]       = float(metrics.box.map)
        results["box_precision"] = float(metrics.box.mp)
        results["box_recall"]    = float(metrics.box.mr)
    if hasattr(metrics, "seg"):
        results["seg_map50"]     = float(metrics.seg.map50)
        results["seg_map"]       = float(metrics.seg.map)
        results["seg_precision"] = float(metrics.seg.mp)
        results["seg_recall"]    = float(metrics.seg.mr)

    return results


# ──────────────────────────────────────────────────────────────────────────────
# Custom inference + IoU computation on a directory
# ──────────────────────────────────────────────────────────────────────────────

def run_inference_on_dir(
    checkpoint: str,
    images_dir: Path,
    masks_dir: Path | None,
    conf: float,
    iou_thresh: float,
    device: str,
    output_dir: Path,
    n_vis: int,
    beta: float
) -> dict:
    """
    Run YOLOv8-seg inference on all images in `images_dir`.
    If `masks_dir` is provided, compute per-image mask IoU and F-beta score.
    """
    from ultralytics import YOLO

    model = YOLO(checkpoint)
    vis_dir = output_dir / "visualizations"
    vis_dir.mkdir(parents=True, exist_ok=True)

    exts  = {".png", ".jpg", ".jpeg"}
    imgs  = sorted(p for p in images_dir.rglob("*") if p.suffix.lower() in exts)

    if not imgs:
        print(f"  No images found in {images_dir}")
        return {}

    per_image_ious = []
    per_image_prec = []
    per_image_rec  = []
    vis_count = 0

    for img_path in imgs:
        # Run inference
        result = model.predict(
            source=str(img_path),
            conf=conf,
            iou=iou_thresh,
            device=device,
            verbose=False,
            retina_masks=True
        )[0]

        h, w = result.orig_shape

        # Build predicted binary mask (union of all detections)
        pred_mask = np.zeros((h, w), dtype=np.uint8)
        if result.masks is not None and len(result.masks.data) > 0:
            for m in result.masks.data.cpu().numpy():
                m_resized = (m > 0.5).astype(np.uint8) * 255
                import cv2
                m_resized = cv2.resize(m_resized, (w, h), interpolation=cv2.INTER_NEAREST)
                pred_mask = np.maximum(pred_mask, m_resized)

        # Compare with GT mask if available
        if masks_dir:
            gt_path = masks_dir / img_path.name
            if gt_path.exists():
                gt = np.array(Image.open(gt_path).convert("L"))
                iou_val = mask_iou(pred_mask, gt)
                per_image_ious.append(iou_val)

                # Compute precision/recall from pixel masks
                tp = ((pred_mask > 0) & (gt > 0)).sum()
                fp = ((pred_mask > 0) & (gt == 0)).sum()
                fn = ((pred_mask == 0) & (gt > 0)).sum()
                prec = float(tp / (tp + fp + 1e-9))
                rec  = float(tp / (tp + fn + 1e-9))
                per_image_prec.append(prec)
                per_image_rec.append(rec)

        # Save visualization
        if vis_count < n_vis:
            _save_vis(img_path, pred_mask, result, vis_dir, vis_count)
            vis_count += 1

    # Aggregate metrics
    metrics = {}
    if per_image_ious:
        mean_iou   = float(np.mean(per_image_ious))
        mean_prec  = float(np.mean(per_image_prec))
        mean_rec   = float(np.mean(per_image_rec))
        f2         = f_beta_score(mean_prec, mean_rec, beta)
        metrics = {
            "mean_mask_iou":  mean_iou,
            "mean_precision": mean_prec,
            "mean_recall":    mean_rec,
            f"f{beta:.0f}_score": f2
        }

    return metrics


def _save_vis(
    img_path: Path,
    pred_mask: np.ndarray,
    result,
    vis_dir: Path,
    idx: int
):
    """Save side-by-side: original | prediction overlay."""
    try:
        import cv2

        orig = np.array(Image.open(img_path).convert("RGB"))
        h, w = orig.shape[:2]

        # Overlay
        overlay = orig.copy().astype(np.float32)
        if pred_mask.any():
            mask_resized = cv2.resize(pred_mask, (w, h), interpolation=cv2.INTER_NEAREST)
            binary = mask_resized > 0
            overlay[binary] = overlay[binary] * 0.55 + np.array([255, 60, 60]) * 0.45

        # Draw bounding boxes + confidence
        if result.boxes is not None:
            for box, conf_score in zip(result.boxes.xyxy.cpu().numpy(),
                                       result.boxes.conf.cpu().numpy()):
                x1, y1, x2, y2 = box.astype(int)
                cv2.rectangle(overlay.astype(np.uint8), (x1, y1), (x2, y2),
                              (255, 255, 0), 2)
                cv2.putText(overlay.astype(np.uint8),
                            f"rip {conf_score:.2f}",
                            (x1, max(y1-5, 10)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)

        side_by_side = np.concatenate([orig, np.clip(overlay, 0, 255).astype(np.uint8)], axis=1)
        Image.fromarray(side_by_side).save(vis_dir / f"pred_{idx:04d}_{img_path.stem}.png")
    except Exception as e:
        pass  # Don't crash on visualization failures


# ──────────────────────────────────────────────────────────────────────────────
# Summary report
# ──────────────────────────────────────────────────────────────────────────────

def print_report(official: dict, custom: dict, beta: float, output_dir: Path):
    print()
    print("=" * 60)
    print("  RipVIS Evaluation Report")
    print("=" * 60)

    if official:
        print("\n  YOLO Official Metrics (COCO-style):")
        for k, v in official.items():
            print(f"    {k:25s}: {v:.4f}")

    if custom:
        print(f"\n  Per-Pixel Mask Metrics:")
        for k, v in custom.items():
            print(f"    {k:25s}: {v:.4f}")

        f2 = custom.get(f"f{beta:.0f}_score", None)
        if f2 is not None:
            rating = "Excellent" if f2 > 0.7 else "Good" if f2 > 0.5 else "Fair" if f2 > 0.3 else "Poor"
            print(f"\n  F{beta:.0f} Score: {f2:.4f}  [{rating}]")
            print(f"  (F2 weights recall 4× precision — missing a rip is penalized heavily)")

    # Save JSON report
    report = {"official_metrics": official, "custom_metrics": custom}
    report_path = output_dir / "eval_report.json"
    import json
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\n  Full report saved: {report_path}")
    if (output_dir / "official_val").is_dir():
        print(f"  Plots saved:       {output_dir}/official_val/")


# ──────────────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()

    print("=" * 60)
    print("  RipVIS — Model Evaluation")
    print("=" * 60)
    print(f"  Checkpoint: {args.checkpoint}")
    print(f"  Conf:       {args.conf}")
    print(f"  IoU thr:    {args.iou}")
    print(f"  Beta:       {args.beta} (F{args.beta:.0f} score)")

    ckpt       = args.checkpoint
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Try CUDA, fall back
    device = args.device
    try:
        import torch
        if device == "0" and not torch.cuda.is_available():
            device = "cpu"
            print("  GPU unavailable, using CPU")
    except ImportError:
        device = "cpu"

    official_metrics = {}
    custom_metrics   = {}

    # ── Official YOLO val ─────────────────────────────────────────────────
    data_yaml = Path(args.data)
    if data_yaml.exists():
        print(f"\n[1/2] Running official YOLO validation on: {data_yaml}")
        try:
            official_metrics = run_official_val(
                ckpt, str(data_yaml), device,
                args.conf, args.iou, output_dir
            )
        except Exception as e:
            print(f"  Official val failed: {e}")
    else:
        print(f"  data.yaml not found ({data_yaml}), skipping official val.")

    # ── Custom inference (with per-pixel IoU) ────────────────────────────
    images_dir = None
    if args.images:
        images_dir = Path(args.images)
    elif data_yaml.exists():
        import yaml
        with open(data_yaml) as f:
            cfg = yaml.safe_load(f)
        base = Path(cfg.get("path", data_yaml.parent))
        split_rel = cfg.get(args.split, f"{args.split}/images")
        images_dir = base / split_rel
        masks_dir = images_dir.parent.parent / args.split / "masks" \
            if not args.masks else Path(args.masks)

    if images_dir and images_dir.is_dir():
        print(f"\n[2/2] Running custom inference on: {images_dir}")
        gt_masks = Path(args.masks) if args.masks else \
                   images_dir.parent.parent / args.split / "masks"
        has_gt = gt_masks.is_dir()
        if not has_gt:
            # Try the synthetic masks directory
            for candidate in [
                Path("data/synthetic/test_synthetic/imprinted/masks"),
                Path("data/test_imprint_out/test_output/masks"),
                images_dir.parent.parent / "masks",
            ]:
                if candidate.is_dir():
                    gt_masks = candidate
                    has_gt = True
                    break

        print(f"  GT masks: {gt_masks if has_gt else 'not found (IoU will be skipped)'}")
        custom_metrics = run_inference_on_dir(
            ckpt, images_dir,
            gt_masks if has_gt else None,
            args.conf, args.iou, device,
            output_dir, args.n_vis, args.beta
        )
    else:
        print("  No image directory found for inference.")

    print_report(official_metrics, custom_metrics, args.beta, output_dir)


if __name__ == "__main__":
    main()
