# RipVIS — Rip Currents Video Instance Segmentation

> **Based on:** [arXiv:2504.01128](https://arxiv.org/abs/2504.01128) — *RipVIS: Rip Currents Video Instance Segmentation Benchmark for Beach Monitoring and Safety* (CVPR 2025)

This repository replicates and extends the [Irikos/rip_currents](https://github.com/Irikos/rip_currents) codebase. It includes:
- Full project structure mirroring the original benchmark
- **Synthetic dataset generation pipeline** using:
  - **Imprinting technique** (copy-paste augmentation + Poisson blending)
  - **Diffusion-based generation** (Stable Diffusion inpainting)
  - **Combined pipeline** for maximum data diversity

For the original benchmark, dataset, models, and results visit: [https://ripvis.ai](https://ripvis.ai)

---

## 📁 Repository Structure

```
RipVIS_project/
├── README.md
├── requirements.txt
├── configs/
│   └── generation_config.yaml      # All tunable parameters
├── data/
│   ├── raw/                        # Real beach frames (from RipVIS or custom)
│   ├── masks/                      # Binary rip current segmentation masks
│   ├── samples/                    # Downloaded sample frames from HuggingFace
│   ├── synthetic/                  # Generated synthetic images (output)
│   └── annotations/                # COCO-format JSON annotations (output)
├── scripts/
│   ├── download_samples.py         # Pull sample frames from HuggingFace
│   ├── prepare_masks.py            # Binarize / pre-process masks
│   ├── imprint_rip.py              # Imprinting pipeline (copy-paste + blending)
│   ├── diffusion_generate.py       # Stable Diffusion inpainting pipeline
│   └── build_dataset.py            # Master pipeline — combines all methods
├── utils/
│   ├── mask_utils.py               # Mask I/O and morphological helpers
│   ├── blend_utils.py              # Poisson blending, alpha compositing
│   └── coco_utils.py               # COCO annotation generation
├── models/                         # Cached diffusion model weights
├── notebooks/
│   ├── 01_explore_data.ipynb       # Dataset exploration and visualization
│   ├── 02_diffusion_pipeline.ipynb # Interactive diffusion generation demo
│   └── 03_evaluate_synthetic.ipynb # FID, visual quality, mask consistency
└── .gitignore
```

---

## 🚀 Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Download Sample Frames (from HuggingFace RipVIS dataset)
```bash
python scripts/download_samples.py --n_samples 20 --output data/samples/
```

### 3. Generate Synthetic Data — Imprinting Only
```bash
python scripts/imprint_rip.py \
    --source_dir data/samples/ \
    --mask_dir data/masks/ \
    --output_dir data/synthetic/ \
    --n_augmentations 5
```

### 4. Generate Synthetic Data — Diffusion Inpainting
```bash
python scripts/diffusion_generate.py \
    --input_dir data/samples/ \
    --mask_dir data/masks/ \
    --output_dir data/synthetic/ \
    --model stabilityai/stable-diffusion-2-inpainting \
    --steps 30
```

### 5. Build Full Dataset (Combined Pipeline)
```bash
python scripts/build_dataset.py \
    --config configs/generation_config.yaml \
    --output data/annotations/synthetic_dataset.json
```

---

## 📄 Paper Reference

```bibtex
@InProceedings{dumitriu2025ripvis,
  author    = {Dumitriu, Andrei and Tatui, Florin and Miron, Florin and Ralhan, Aakash and Ionescu, Radu Tudor and Timofte, Radu},
  title     = {RipVIS: Rip Currents Video Instance Segmentation Benchmark for Beach Monitoring and Safety},
  booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  year      = {2025},
  doi       = {10.1109/CVPR52734.2025.00325}
}
```

---

## 🔬 Synthetic Data Generation — Methodology

### Method A: Imprinting (Copy-Paste + Blending)
1. Extract rip current region from an annotated source frame using its binary mask
2. Apply geometric augmentations (flip, scale, rotate) to the extracted region
3. Poisson-blend the rip current texture onto a target background beach image
4. Record the new bounding polygon for COCO annotation

### Method B: Diffusion-Based Generation
1. Take a clean beach image (no rip current)
2. Draw / load a binary mask defining where a rip current should appear
3. Run `stable-diffusion-2-inpainting` with a rip-current-specific prompt
4. The model fills the masked region with a photorealistic rip current
5. Use the input mask as the segmentation annotation

### Method C: Combined Pipeline (Recommended)
1. **Imprint** first → creates a structurally-valid composite
2. **Diffusion refine** → diffuses the boundary seam for photorealism
3. Produces the highest-quality synthetic training pairs

---

## 📊 Models (from original RipVIS benchmark)

| Model             | Backbone     | Notes                     |
|-------------------|--------------|---------------------------|
| Mask R-CNN        | ResNet-50    | Baseline instance seg     |
| Cascade Mask R-CNN| ResNet-101   | Higher AP at higher IoU   |
| SparseInst        | ResNet-50    | Fast, real-time capable   |
| YOLO11            | CSPDarkNet   | State-of-the-art speed    |

Post-processing: **Temporal Confidence Aggregation (TCA)** to reduce false negatives across video frames.

---

## 🌊 Dataset
The RipVIS dataset (real annotated data) is hosted at:
- **HuggingFace:** [Irikos/RipVIS](https://huggingface.co/datasets/Irikos/RipVIS)
- **Website:** [https://ripvis.ai](https://ripvis.ai)
