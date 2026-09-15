import torch
from diffusers import AutoencoderKL
from transformers import CLIPTextModel, CLIPTokenizer

print("Testing model load...")
try:
    vae = AutoencoderKL.from_pretrained("runwayml/stable-diffusion-v1-5", subfolder="vae")
    print("VAE loaded successfully!")
    tokenizer = CLIPTokenizer.from_pretrained("runwayml/stable-diffusion-v1-5", subfolder="tokenizer")
    text_encoder = CLIPTextModel.from_pretrained("runwayml/stable-diffusion-v1-5", subfolder="text_encoder")
    print("Text encoder loaded successfully!")
except Exception as e:
    print("Error:", e)
