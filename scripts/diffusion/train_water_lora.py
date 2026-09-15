#!/usr/bin/env python3
"""
train_water_lora.py
────────────────────
Stage 2: Fine-Tuning with LoRA for Water-Specialized Diffusion.

Specifications:
- Base Model: runwayml/stable-diffusion-v1-5
- Target Modules: Attention ("to_q", "to_k", "to_v", "to_out.0") & Feed-Forward ("ff.net.0.proj", "ff.net.2")
- Trigger Words: "underwater", "caustics", "splash", "aerial view", "water surface", "ripples", "turbulent wake"
- Training: Directly on pre-computed VAE latents [16, 4, 64, 64] and text embeddings [77, 768]
- Multi-epoch training with validation loss tracking and standalone adapter export.
"""

import argparse
import glob
import json
import logging
import math
import os
import random
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from diffusers import UNet2DConditionModel, DDPMScheduler
from peft import LoraConfig, get_peft_model
from transformers import CLIPTokenizer, CLIPTextModel

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger("TrainWaterLoRA")

WATER_TRIGGERS = [
    "underwater",
    "caustics",
    "splash",
    "aerial view",
    "water surface",
    "ripples",
    "turbulent wake",
]


class PrecomputedWaterDataset(Dataset):
    """
    Loads pre-computed VAE latents [16, 4, 64, 64] and text embeddings [77, 768].
    Flattens the 16 temporal frames into individual training instances,
    with random horizontal flip augmentation in latent space.
    """
    def __init__(
        self,
        latents_dir: Path,
        embeddings_dir: Path,
        annotations_file: Optional[Path] = None,
        tokenizer: Optional[CLIPTokenizer] = None,
        text_encoder: Optional[CLIPTextModel] = None,
        device: Optional[torch.device] = None,
        random_flip: bool = True,
    ):
        self.random_flip = random_flip
        self.samples: List[Tuple[torch.Tensor, torch.Tensor, str]] = []

        latent_files = sorted(list(latents_dir.glob("*.pt")))
        logger.info(f"Loading pre-computed latents from {latents_dir} ({len(latent_files)} video clips)...")

        # Load trigger-enriched annotations if available
        trigger_lookup = {}
        if annotations_file and annotations_file.exists():
            with open(annotations_file, "r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        entry = json.loads(line.strip())
                        vid_id = entry.get("video_id")
                        prompt = entry.get("prompt", "")
                        # Determine relevant trigger words based on metadata / regime
                        regime = entry.get("fluid_regime", "").lower()
                        cam = entry.get("camera_dynamics", "").lower()
                        matched_triggers = []

                        if "underwater" in regime or "reef" in regime or "scuba" in regime:
                            matched_triggers.extend(["underwater", "caustics", "water surface"])
                        if "splash" in regime or "plunge" in regime or "dive" in regime or "jump" in regime:
                            matched_triggers.extend(["splash", "turbulent wake", "water surface", "ripples"])
                        if "aerial" in cam or "drone" in cam or "top_down" in cam:
                            matched_triggers.extend(["aerial view", "water surface", "ripples", "turbulent wake"])
                        if "wake" in regime or "freestyle" in regime or "breaststroke" in regime:
                            matched_triggers.extend(["turbulent wake", "ripples", "water surface"])

                        if not matched_triggers:
                            matched_triggers = ["water surface", "ripples", "caustics"]

                        # Deduplicate while preserving order
                        unique_triggers = list(dict.fromkeys(matched_triggers))
                        trigger_prefix = ", ".join(unique_triggers)
                        enriched_prompt = f"{trigger_prefix}, {prompt}"
                        trigger_lookup[vid_id] = (enriched_prompt, unique_triggers)

        # Cache samples
        for lat_path in latent_files:
            vid_id = lat_path.stem
            emb_path = embeddings_dir / f"{vid_id}.pt"
            if not emb_path.exists():
                continue

            lat_data = torch.load(lat_path, map_location="cpu", weights_only=False)
            emb_data = torch.load(emb_path, map_location="cpu", weights_only=False)

            latents_16f = lat_data["latents"]  # [16, 4, 64, 64]
            dense_embeds = emb_data["dense_embeds"]  # [77, 768]

            # If trigger enrichment requested and text encoder provided, re-encode with trigger prefix
            if vid_id in trigger_lookup and tokenizer is not None and text_encoder is not None and device is not None:
                enriched_text = trigger_lookup[vid_id][0]
                with torch.no_grad():
                    toks = tokenizer(
                        enriched_text,
                        padding="max_length",
                        max_length=tokenizer.model_max_length,
                        truncation=True,
                        return_tensors="pt"
                    ).input_ids.to(device)
                    dense_embeds = text_encoder(toks)[0].squeeze(0).half().cpu()

            # Add each of the 16 frames as a training sample
            for f_idx in range(latents_16f.shape[0]):
                frame_latent = latents_16f[f_idx]  # [4, 64, 64]
                self.samples.append((frame_latent, dense_embeds, vid_id))

        logger.info(f"Initialized dataset with {len(self.samples)} latent frame instances.")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        latent, embed, vid_id = self.samples[idx]
        latent = latent.clone()

        # Random horizontal flip in latent space: width dimension is axis 2
        if self.random_flip and random.random() < 0.5:
            latent = torch.flip(latent, dims=[2])

        return {
            "latent": latent.float(),
            "encoder_hidden_states": embed.float(),
            "video_id": vid_id,
        }


def evaluate_val_loss(
    unet: torch.nn.Module,
    noise_scheduler: DDPMScheduler,
    val_loader: DataLoader,
    device: torch.device,
    weight_dtype: torch.dtype,
    max_batches: int = 25,
) -> float:
    """Evaluates validation loss on held-out latent frames."""
    unet.eval()
    val_losses = []

    with torch.no_grad():
        for b_idx, batch in enumerate(val_loader):
            if b_idx >= max_batches:
                break
            latents = batch["latent"].to(device, dtype=weight_dtype)
            encoder_hidden_states = batch["encoder_hidden_states"].to(device, dtype=weight_dtype)

            noise = torch.randn_like(latents)
            bsz = latents.shape[0]
            timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (bsz,), device=device).long()
            noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

            model_pred = unet(noisy_latents, timesteps, encoder_hidden_states).sample
            target = noise if noise_scheduler.config.prediction_type == "epsilon" else noise_scheduler.get_velocity(latents, noise, timesteps)
            loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")
            val_losses.append(loss.item())

    unet.train()
    return float(np.mean(val_losses)) if val_losses else 0.0


def main():
    parser = argparse.ArgumentParser(description="Stage 2: Fine-Tuning with LoRA for Water-Specialized Diffusion.")
    parser.add_argument("--base_model", type=str, default="runwayml/stable-diffusion-v1-5")
    parser.add_argument("--data_dir", type=str, default="data/video_diffusion")
    parser.add_argument("--output_dir", type=str, default="models/water_swimmer_lora")
    parser.add_argument("--lora_rank", type=int, default=32)
    parser.add_argument("--lora_alpha", type=int, default=64)
    parser.add_argument("--lora_dropout", type=float, default=0.05)
    parser.add_argument("--train_batch_size", type=int, default=2)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4)
    parser.add_argument("--learning_rate", type=float, default=8.0e-5)
    parser.add_argument("--lr_warmup_steps", type=int, default=50)
    parser.add_argument("--max_train_steps", type=int, default=1000)
    parser.add_argument("--checkpointing_steps", type=int, default=250)
    parser.add_argument("--eval_steps", type=int, default=250)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    weight_dtype = torch.float16 if device.type == "cuda" else torch.float32
    logger.info(f"Starting Stage 2 LoRA Training on device: {device} ({torch.cuda.get_device_name(0) if device.type == 'cuda' else 'CPU'})")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    base_data_dir = Path(args.data_dir)

    # 1. Load Tokenizer & Text Encoder for Trigger Enrichment
    logger.info(f"Loading CLIP Tokenizer & Text Encoder: {args.base_model}...")
    tokenizer = CLIPTokenizer.from_pretrained(args.base_model, subfolder="tokenizer")
    text_encoder = CLIPTextModel.from_pretrained(args.base_model, subfolder="text_encoder", torch_dtype=weight_dtype).to(device)
    text_encoder.eval()
    text_encoder.requires_grad_(False)

    # 2. Build Datasets
    ann_train = base_data_dir / "annotations" / "metadata_train.jsonl"
    ann_val = base_data_dir / "annotations" / "metadata_val.jsonl"

    train_dataset = PrecomputedWaterDataset(
        latents_dir=base_data_dir / "latents" / "train",
        embeddings_dir=base_data_dir / "embeddings" / "train",
        annotations_file=ann_train,
        tokenizer=tokenizer,
        text_encoder=text_encoder,
        device=device,
        random_flip=True,
    )

    val_dataset = PrecomputedWaterDataset(
        latents_dir=base_data_dir / "latents" / "val",
        embeddings_dir=base_data_dir / "embeddings" / "val",
        annotations_file=ann_val,
        tokenizer=tokenizer,
        text_encoder=text_encoder,
        device=device,
        random_flip=False,
    )

    # Free text_encoder from GPU to maximize training VRAM
    del text_encoder
    torch.cuda.empty_cache()

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.train_batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.train_batch_size,
        shuffle=False,
        num_workers=0,
    )

    # 3. Load UNet & Inject LoRA on Target Modules
    logger.info(f"Loading UNet2DConditionModel from {args.base_model} in {weight_dtype}...")
    unet = UNet2DConditionModel.from_pretrained(
        args.base_model,
        subfolder="unet",
        torch_dtype=weight_dtype,
    )
    unet.requires_grad_(False)

    # Target both Attention and Feed-Forward submodules
    target_modules = ["to_q", "to_k", "to_v", "to_out.0", "ff.net.0.proj", "ff.net.2"]
    logger.info(f"Injecting LoRA into Attention & Feed-Forward target modules: {target_modules}")

    lora_config = LoraConfig(
        r=args.lora_rank,
        lora_alpha=args.lora_alpha,
        init_lora_weights="gaussian",
        target_modules=target_modules,
        lora_dropout=args.lora_dropout,
    )
    unet = get_peft_model(unet, lora_config)
    unet.print_trainable_parameters()

    # Enable gradient checkpointing to reduce activation memory
    if hasattr(unet, "enable_gradient_checkpointing"):
        unet.enable_gradient_checkpointing()

    unet.to(device)


    # 4. Noise Scheduler & Optimizer
    noise_scheduler = DDPMScheduler.from_pretrained(args.base_model, subfolder="scheduler")

    from transformers import get_cosine_schedule_with_warmup
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, unet.parameters()),
        lr=args.learning_rate,
        betas=(0.9, 0.999),
        weight_decay=1e-2,
        eps=1e-08,
    )
    lr_scheduler = get_cosine_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps,
        num_training_steps=args.max_train_steps,
    )
    scaler = torch.amp.GradScaler("cuda", enabled=(device.type == "cuda"))

    # 5. Training Loop
    logger.info(f"\n=======================================================")
    logger.info(f"Training Plan: {args.max_train_steps} steps")
    logger.info(f"Batch: {args.train_batch_size} x Grad Accum: {args.gradient_accumulation_steps} (Effective Batch: {args.train_batch_size * args.gradient_accumulation_steps})")
    logger.info(f"Dataset Size: {len(train_dataset)} latent instances ({len(train_dataset)//16} clips x 16 frames)")
    logger.info(f"Learning Rate: {args.learning_rate} (Cosine schedule with {args.lr_warmup_steps} warmup steps)")
    logger.info(f"Water Trigger Words Active: {WATER_TRIGGERS}")
    logger.info(f"=======================================================\n")

    global_step = 0
    train_loss_history = []
    val_loss_history = []
    start_time = time.time()

    unet.train()
    epochs = math.ceil(args.max_train_steps / (len(train_loader) / args.gradient_accumulation_steps)) + 2

    for epoch in range(epochs):
        epoch_loss = []
        for step, batch in enumerate(train_loader):
            if global_step >= args.max_train_steps:
                break

            latents = batch["latent"].to(device, dtype=weight_dtype)
            encoder_hidden_states = batch["encoder_hidden_states"].to(device, dtype=weight_dtype)

            # Sample noise and random timesteps
            noise = torch.randn_like(latents)
            bsz = latents.shape[0]
            timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (bsz,), device=device).long()
            noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

            with torch.amp.autocast("cuda", enabled=(device.type == "cuda")):
                model_pred = unet(noisy_latents, timesteps, encoder_hidden_states).sample
                target = noise if noise_scheduler.config.prediction_type == "epsilon" else noise_scheduler.get_velocity(latents, noise, timesteps)
                loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")
                loss = loss / args.gradient_accumulation_steps

            scaler.scale(loss).backward()
            epoch_loss.append(loss.item() * args.gradient_accumulation_steps)

            if (step + 1) % args.gradient_accumulation_steps == 0 or (step + 1) == len(train_loader):
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(unet.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                lr_scheduler.step()
                global_step += 1

                step_loss = float(np.mean(epoch_loss[-args.gradient_accumulation_steps:]))
                train_loss_history.append({"step": global_step, "loss": round(step_loss, 4), "lr": float(lr_scheduler.get_last_lr()[0])})

                if global_step % 50 == 0:
                    elapsed = time.time() - start_time
                    steps_per_sec = global_step / elapsed if elapsed > 0 else 0
                    logger.info(f"Epoch {epoch+1} | Step {global_step}/{args.max_train_steps} | Loss: {step_loss:.4f} | Speed: {steps_per_sec:.2f} step/s")

                # Validation Evaluation
                if global_step % args.eval_steps == 0:
                    v_loss = evaluate_val_loss(unet, noise_scheduler, val_loader, device, weight_dtype)
                    val_loss_history.append({"step": global_step, "val_loss": round(v_loss, 4)})
                    logger.info(f"--- Step {global_step} Evaluation: Validation Loss = {v_loss:.4f} ---")

                # Periodic Checkpoint
                if global_step % args.checkpointing_steps == 0:
                    ckpt_dir = output_dir / f"checkpoint-{global_step}"
                    logger.info(f"Saving checkpoint to: {ckpt_dir}...")
                    unet.save_pretrained(str(ckpt_dir))

        if global_step >= args.max_train_steps:
            break

    total_time = time.time() - start_time
    logger.info(f"LoRA training finished in {total_time/60:.2f} minutes!")

    # 6. Save Standalone Final LoRA
    final_lora_dir = output_dir / "final_lora"
    logger.info(f"Saving final standalone LoRA adapter to: {final_lora_dir}...")
    unet.save_pretrained(str(final_lora_dir))

    # 7. Export Training Report
    metrics_dir = Path("runs/training_metrics")
    metrics_dir.mkdir(parents=True, exist_ok=True)
    report = {
        "model": "runwayml/stable-diffusion-v1-5",
        "lora_rank": args.lora_rank,
        "lora_alpha": args.lora_alpha,
        "target_modules": target_modules,
        "water_trigger_words": WATER_TRIGGERS,
        "total_train_steps": global_step,
        "total_epochs": epoch + 1,
        "training_time_minutes": round(total_time / 60, 2),
        "final_train_loss": round(train_loss_history[-1]["loss"], 4) if train_loss_history else 0.0,
        "final_val_loss": round(val_loss_history[-1]["val_loss"], 4) if val_loss_history else 0.0,
        "train_loss_curve": train_loss_history[::10],  # Subsample every 10 steps
        "val_loss_curve": val_loss_history,
        "saved_checkpoints": [f"checkpoint-{s}" for s in range(args.checkpointing_steps, args.max_train_steps + 1, args.checkpointing_steps)],
        "final_adapter_path": str(final_lora_dir),
    }

    report_path = metrics_dir / "water_lora_train_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    logger.info(f"Training report saved to: {report_path}")
    print("\nStage 2 Training Completed Successfully!")
    print(f"Final Train Loss: {report['final_train_loss']}, Final Val Loss: {report['final_val_loss']}")


if __name__ == "__main__":
    main()
