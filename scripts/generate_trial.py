"""
generate_trial.py
─────────────────
Generates 5 high-quality trial images of clearly visible swimmers inside and near rip currents
using the fine-tuned LoRA diffusion model (trained up to 800 steps on the AFO swimmer dataset).
Outputs to data/trial/ and creates an evaluation collage.
"""

import argparse
from pathlib import Path
import torch
from PIL import Image, ImageDraw, ImageFont
from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler
from peft import PeftModel


TRIAL_PROMPTS = [
    # 1. Medium-altitude drone view: Freestyle swimmer in rip current channel
    "medium altitude aerial drone photograph 20m above sea of an adult person swimming freestyle in clear turquoise ocean water inside an outgoing rip current channel, visible human head with dark hair and blue swim shorts, arms reaching forward in front crawl, splashing white water wake foam and kicking bubbles, natural ocean caustics, photorealistic 8k",

    # 2. Medium-altitude drone view: Two swimmers near rip current foam neck
    "overhead drone photo 15 meters above water of two swimmers treading water adjacent to a foamy rip current corridor, authentic human body proportions, natural skin tones, visible arm strokes, crystal clear turquoise water, bright sunlight caustics, photorealistic",

    # 3. Medium-altitude drone view: Swimmer caught in breaking surf near rip trench
    "medium altitude top-down drone photograph of an ocean swimmer floating and swimming in turquoise coastal surf near an outgoing rip current trench, visible human head and shoulders, white foam wave splash, realistic ocean optics and water refraction",

    # 4. Medium-altitude drone view: Swimmer navigating seaward in rip channel
    "aerial drone capture 20m altitude of a person swimming breaststroke in emerald sea water alongside a foamy rip current conduit, clearly visible head and arms, authentic water wake ripples and splash bubbles, photorealistic marine photography",

    # 5. Medium-altitude drone view: Swimmer in rip current head
    "overhead aerial photograph of a swimmer in clear turquoise ocean water at the expanding head of a rip current, visible human body anatomy, natural water transparency and wave distortion, high definition Hasselblad"
]

NEGATIVE_PROMPT = (
    "boat, kayak, canoe, surfboard, paddleboard, red hull, vessel, oars, paddles, "
    "cartoon, 3d render, illustration, drawing, painting, bad anatomy, deformed body, "
    "extra limbs, missing limbs, blurry, plastic skin, floating sticker, text, watermark, CGI, ugly"
)


def run_trial_generation(
    output_dir: str = "data/trial",
    lora_path: str = "models/diffusion_rip_swimmer_lora/final_lora"
):
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    images_dir = out_path / "images"
    images_dir.mkdir(parents=True, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    weight_dtype = torch.float16 if device == "cuda" else torch.float32

    print(f"Loading Base SD Model & LoRA ({lora_path})...")
    pipe = StableDiffusionPipeline.from_pretrained(
        "runwayml/stable-diffusion-v1-5",
        torch_dtype=weight_dtype,
        safety_checker=None
    )
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(
        pipe.scheduler.config,
        use_karras_sigmas=True,
        algorithm_type="dpmsolver++"
    )

    if Path(lora_path).exists():
        import json
        from peft import LoraConfig
        cfg_file = Path(lora_path) / "adapter_config.json"
        if cfg_file.exists():
            with open(cfg_file) as f:
                raw_cfg = json.load(f)
            # Filter to only valid LoraConfig fields
            valid_keys = {
                "r", "lora_alpha", "lora_dropout", "target_modules", "bias",
                "fan_in_fan_out", "init_lora_weights", "peft_type"
            }
            clean_cfg = {k: v for k, v in raw_cfg.items() if k in valid_keys and v is not None}
            clean_cfg["peft_type"] = "LORA"
            with open(cfg_file, "w") as f:
                json.dump(clean_cfg, f, indent=2)
                
        pipe.unet = PeftModel.from_pretrained(pipe.unet, lora_path)
        print(f"✓ Successfully loaded fine-tuned LoRA from {lora_path}!")

    pipe.to(device)
    if device == "cuda":
        pipe.enable_attention_slicing()

    generated_images = []
    seeds = [1024, 2048, 3072, 4096, 5120]

    for i, prompt in enumerate(TRIAL_PROMPTS):
        seed = seeds[i]
        generator = torch.Generator(device=device).manual_seed(seed)
        print(f"Generating Trial Scene {i+1}/5 (Seed: {seed})...")
        
        result = pipe(
            prompt=prompt,
            negative_prompt=NEGATIVE_PROMPT,
            num_inference_steps=40,
            guidance_scale=7.5,
            generator=generator,
            height=512,
            width=512
        ).images[0]

        img_file = images_dir / f"trial_sample_{i+1:02d}.jpg"
        result.save(img_file, quality=95)
        generated_images.append((img_file, prompt))
        print(f"  ✓ Saved {img_file}")

    # Build 5-sample evaluation collage
    create_trial_collage(generated_images, out_path / "trial_5_samples.jpg")
    print("✓ 5 trial images generated and saved in data/trial/!")


def create_trial_collage(img_tuples, output_path: Path):
    from PIL import ImageDraw
    card_w, card_h = 480, 480
    header_h = 42
    
    canvas_w = card_w * 3 + 40
    canvas_h = (card_h + header_h) * 2 + 30
    canvas = Image.new("RGB", (canvas_w, canvas_h), (20, 24, 30))
    draw = ImageDraw.Draw(canvas)

    coords = [
        (10, 10),
        (card_w + 20, 10),
        (card_w * 2 + 30, 10),
        (int(canvas_w / 2 - card_w - 5), card_h + header_h + 20),
        (int(canvas_w / 2 + 5), card_h + header_h + 20)
    ]

    labels = [
        "Sample 1: Swimmer in Rip Current Channel",
        "Sample 2: Two Swimmers Near Rip Foam Plume",
        "Sample 3: Swimmer in Rip Current Trench Boundary",
        "Sample 4: Freestyle Swimmer in Coastal Ocean",
        "Sample 5: Swimmer in Offshore Rip Current Head"
    ]

    for idx, ((img_path, _), (x, y), label) in enumerate(zip(img_tuples, coords, labels)):
        img = Image.open(img_path).convert("RGB").resize((card_w, card_h))
        draw.rectangle([(x, y), (x + card_w, y + header_h - 4)], fill=(34, 40, 52))
        draw.text((x + 12, y + 12), f"[{idx+1}] {label}", fill=(245, 245, 245))
        canvas.paste(img, (x, y + header_h))

    canvas.save(output_path, quality=95)
    print(f"✓ Trial collage saved to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", type=str, default="data/trial")
    parser.add_argument("--lora_path", type=str, default="models/diffusion_rip_swimmer_lora/final_lora")
    args = parser.parse_args()
    run_trial_generation(args.output_dir, args.lora_path)
