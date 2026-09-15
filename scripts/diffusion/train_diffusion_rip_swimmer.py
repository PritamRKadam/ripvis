"""
train_diffusion_rip_swimmer.py
──────────────────────────────
Fine-tunes a Diffusion Model (LoRA on Stable Diffusion) for generating
photorealistic rip currents and swimmers in surf zones.

Key Features:
- Memory-efficient LoRA training (compatible with 8GB-12GB GPUs like RTX 3060)
- Mixed precision (fp16) + Gradient Checkpointing
- Dataset loading from metadata.jsonl (images + descriptive captions)
- Checkpoint saving and standalone LoRA adapter export

Usage:
    python scripts/train_diffusion_rip_swimmer.py \
        --config configs/diffusion_training_config.yaml
"""

import argparse
import gc
import json
import math
import os
import sys
from pathlib import Path
from typing import Optional

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from tqdm import tqdm
import yaml

from transformers import CLIPTokenizer, CLIPTextModel
from diffusers import (
    AutoencoderKL,
    UNet2DConditionModel,
    DDPMScheduler,
)
from peft import LoraConfig, get_peft_model


class RipSwimmerDataset(Dataset):
    def __init__(self, data_dir: str, metadata_file: str, tokenizer, size: int = 512, random_flip: bool = True):
        self.data_dir = Path(data_dir)
        self.tokenizer = tokenizer
        self.size = size
        self.entries = []

        with open(metadata_file, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    self.entries.append(json.loads(line.strip()))

        self.transform = transforms.Compose([
            transforms.Resize((size, size), interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.RandomHorizontalFlip() if random_flip else transforms.Lambda(lambda x: x),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
        ])

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, idx):
        entry = self.entries[idx]
        img_path = self.data_dir / entry["file_name"]
        
        try:
            image = Image.open(img_path).convert("RGB")
        except Exception:
            image = Image.new("RGB", (self.size, self.size), (0, 0, 0))

        pixel_values = self.transform(image)
        prompt = entry.get("text", "aerial view of ocean beach and rip current with swimmers")

        input_ids = self.tokenizer(
            prompt,
            padding="max_length",
            truncation=True,
            max_length=self.tokenizer.model_max_length,
            return_tensors="pt",
        ).input_ids[0]

        return {
            "pixel_values": pixel_values,
            "input_ids": input_ids
        }


def parse_args():
    p = argparse.ArgumentParser(description="Train LoRA for Rip Current & Swimmer Diffusion.")
    p.add_argument("--config", type=str, default="configs/diffusion_training_config.yaml",
                   help="Path to YAML configuration file.")
    p.add_argument("--pretrained_model_name_or_path", type=str, default="runwayml/stable-diffusion-v1-5",
                   help="Base model path or HF repo ID.")
    p.add_argument("--data_dir", type=str, default="data/diffusion_train",
                   help="Directory containing training images and metadata.jsonl.")
    p.add_argument("--output_dir", type=str, default="models/diffusion_rip_swimmer_lora",
                   help="Directory to save LoRA checkpoints.")
    p.add_argument("--resolution", type=int, default=512)
    p.add_argument("--train_batch_size", type=int, default=4)
    p.add_argument("--gradient_accumulation_steps", type=int, default=2)
    p.add_argument("--learning_rate", type=float, default=8.0e-5)
    p.add_argument("--lr_warmup_steps", type=int, default=50)
    p.add_argument("--max_train_steps", type=int, default=800)
    p.add_argument("--checkpointing_steps", type=int, default=200)
    p.add_argument("--lora_rank", type=int, default=32)
    p.add_argument("--lora_alpha", type=int, default=64)
    p.add_argument("--resume_from_checkpoint", type=str, default=None,
                   help="Path to previous LoRA checkpoint to resume/continue training from.")
    p.add_argument("--mixed_precision", type=str, default="fp16", choices=["no", "fp16", "bf16"])
    p.add_argument("--dataloader_num_workers", type=int, default=2)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def train(args):
    # Load config if present
    if args.config and Path(args.config).exists():
        with open(args.config, "r") as f:
            cfg = yaml.safe_load(f)
            if "model" in cfg:
                args.pretrained_model_name_or_path = cfg["model"].get("pretrained_model_name_or_path", args.pretrained_model_name_or_path)
                args.resolution = cfg["model"].get("resolution", args.resolution)
            if "lora" in cfg:
                args.lora_rank = cfg["lora"].get("r", args.lora_rank)
                args.lora_alpha = cfg["lora"].get("lora_alpha", args.lora_alpha)
                args.target_modules = cfg["lora"].get("target_modules", getattr(args, "target_modules", ["to_k", "to_q", "to_v", "to_out.0", "ff.net.0.proj", "ff.net.2"]))
            if "training" in cfg:
                t_cfg = cfg["training"]
                args.output_dir = t_cfg.get("output_dir", args.output_dir)
                args.train_batch_size = t_cfg.get("train_batch_size", args.train_batch_size)
                args.gradient_accumulation_steps = t_cfg.get("gradient_accumulation_steps", args.gradient_accumulation_steps)
                args.learning_rate = float(t_cfg.get("learning_rate", args.learning_rate))
                args.lr_warmup_steps = t_cfg.get("lr_warmup_steps", args.lr_warmup_steps)
                args.max_train_steps = t_cfg.get("max_train_steps", args.max_train_steps)
                args.checkpointing_steps = t_cfg.get("checkpointing_steps", args.checkpointing_steps)
                args.mixed_precision = t_cfg.get("mixed_precision", args.mixed_precision)
                args.dataloader_num_workers = t_cfg.get("dataloader_num_workers", args.dataloader_num_workers)

    if not hasattr(args, "target_modules"):
        args.target_modules = ["to_k", "to_q", "to_v", "to_out.0", "ff.net.0.proj", "ff.net.2"]

    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using Compute Device: {device} ({torch.cuda.get_device_name(0) if device.type == 'cuda' else 'CPU'})")

    weight_dtype = torch.float16 if (args.mixed_precision == "fp16" and device.type == "cuda") else (torch.bfloat16 if (args.mixed_precision == "bf16" and device.type == "cuda") else torch.float32)

    os.makedirs(args.output_dir, exist_ok=True)

    print(f"Loading Base Diffusion Model: {args.pretrained_model_name_or_path}...")
    tokenizer = CLIPTokenizer.from_pretrained(args.pretrained_model_name_or_path, subfolder="tokenizer")
    text_encoder = CLIPTextModel.from_pretrained(args.pretrained_model_name_or_path, subfolder="text_encoder", torch_dtype=weight_dtype)
    vae = AutoencoderKL.from_pretrained(args.pretrained_model_name_or_path, subfolder="vae", torch_dtype=weight_dtype)
    unet = UNet2DConditionModel.from_pretrained(args.pretrained_model_name_or_path, subfolder="unet")
    noise_scheduler = DDPMScheduler.from_pretrained(args.pretrained_model_name_or_path, subfolder="scheduler")

    # Freeze VAE & Text Encoder
    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)
    unet.requires_grad_(False)

    # Enable SDPA for fast attention
    if hasattr(F, "scaled_dot_product_attention"):
        unet.set_default_attn_processor()

    # Enable gradient checkpointing to reduce memory
    if hasattr(unet, "enable_gradient_checkpointing"):
        unet.enable_gradient_checkpointing()

    # Configure LoRA on UNet
    if args.resume_from_checkpoint and Path(args.resume_from_checkpoint).exists():
        print(f"Resuming LoRA from checkpoint: {args.resume_from_checkpoint}...")
        from peft import PeftModel
        unet = PeftModel.from_pretrained(unet, args.resume_from_checkpoint, is_trainable=True)
    else:
        lora_config = LoraConfig(
            r=args.lora_rank,
            lora_alpha=args.lora_alpha,
            init_lora_weights="gaussian",
            target_modules=args.target_modules,
        )
        unet = get_peft_model(unet, lora_config)
    unet.print_trainable_parameters()

    # Move models to device
    vae.to(device, dtype=weight_dtype)
    text_encoder.to(device, dtype=weight_dtype)
    unet.to(device)

    # Dataset & DataLoader
    metadata_file = Path(args.data_dir) / "metadata.jsonl"
    if not metadata_file.exists():
        raise FileNotFoundError(f"Metadata file not found: {metadata_file}. Run scripts/prepare_diffusion_dataset.py first.")

    train_dataset = RipSwimmerDataset(
        data_dir=args.data_dir,
        metadata_file=str(metadata_file),
        tokenizer=tokenizer,
        size=args.resolution,
    )
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.train_batch_size,
        shuffle=True,
        num_workers=args.dataloader_num_workers,
        pin_memory=False,
    )

    # Optimizer & Scheduler
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

    use_amp = (device.type == "cuda" and args.mixed_precision in ["fp16", "bf16"])
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    print(f"\n────────────────────────────────────────────────────────────────────────")
    print(f"Starting LoRA training for {args.max_train_steps} steps")
    print(f"Batch Size: {args.train_batch_size} × Grad Accum: {args.gradient_accumulation_steps} (Effective: {args.train_batch_size * args.gradient_accumulation_steps})")
    print(f"Learning Rate: {args.learning_rate} (Cosine decay with {args.lr_warmup_steps} warmup steps)")
    print(f"LoRA Rank: {args.lora_rank}, Alpha: {args.lora_alpha}")
    print(f"Checkpoint Interval: every {args.checkpointing_steps} steps")
    print(f"────────────────────────────────────────────────────────────────────────\n", flush=True)

    global_step = 0
    progress_bar = tqdm(total=args.max_train_steps, desc="Diffusion LoRA Training")

    unet.train()
    epochs = math.ceil(args.max_train_steps / (len(train_dataloader) / args.gradient_accumulation_steps)) + 2

    for epoch in range(epochs):
        for step, batch in enumerate(train_dataloader):
            if global_step >= args.max_train_steps:
                break

            pixel_values = batch["pixel_values"].to(device, dtype=weight_dtype)
            input_ids = batch["input_ids"].to(device)

            with torch.amp.autocast("cuda", dtype=weight_dtype, enabled=use_amp):
                # Convert images to latent space using VAE
                with torch.no_grad():
                    latents = vae.encode(pixel_values).latent_dist.sample()
                    latents = latents * vae.config.scaling_factor

                # Sample random noise
                noise = torch.randn_like(latents)
                bsz = latents.shape[0]
                timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (bsz,), device=device).long()

                # Add noise to latents according to schedule
                noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

                # Get text conditioning embeddings
                with torch.no_grad():
                    encoder_hidden_states = text_encoder(input_ids)[0]

                # Predict the noise residual
                model_pred = unet(noisy_latents, timesteps, encoder_hidden_states).sample

                # Compute diffusion MSE loss
                target = noise
                loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")
                loss = loss / args.gradient_accumulation_steps

            if use_amp:
                scaler.scale(loss).backward()
            else:
                loss.backward()

            if (step + 1) % args.gradient_accumulation_steps == 0:
                if use_amp:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(unet.parameters(), 1.0)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    torch.nn.utils.clip_grad_norm_(unet.parameters(), 1.0)
                    optimizer.step()

                lr_scheduler.step()
                optimizer.zero_grad()
                global_step += 1
                progress_bar.update(1)
                current_lr = lr_scheduler.get_last_lr()[0]
                raw_loss = loss.item() * args.gradient_accumulation_steps
                progress_bar.set_postfix({"loss": f"{raw_loss:.4f}", "lr": f"{current_lr:.2e}"})

                if global_step % 50 == 0 or global_step == 1:
                    print(f"Step {global_step:04d}/{args.max_train_steps} - Loss: {raw_loss:.4f} - LR: {current_lr:.2e}", flush=True)
                    gc.collect()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()

                if global_step % args.checkpointing_steps == 0 or global_step == args.max_train_steps:
                    ckpt_dir = Path(args.output_dir) / f"checkpoint-{global_step}"
                    unet.save_pretrained(ckpt_dir)
                    print(f"\nSaved checkpoint to {ckpt_dir}", flush=True)

    # Save final LoRA adapter
    final_dir = Path(args.output_dir) / "final_lora"
    unet.save_pretrained(final_dir)
    print(f"\n✓ Training complete! Final LoRA adapter saved to {final_dir}")


if __name__ == "__main__":
    args = parse_args()
    train(args)
