import sys
import torch
from diffusers import AnimateDiffPipeline, MotionAdapter, DDIMScheduler
from peft import PeftModel

print("1. Loading Motion Adapter...", flush=True)
adapter = MotionAdapter.from_pretrained("guoyww/animatediff-motion-adapter-v1-5-2", torch_dtype=torch.float16)

print("2. Loading Base Model runwayml/stable-diffusion-v1-5...", flush=True)
pipe = AnimateDiffPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5",
    motion_adapter=adapter,
    torch_dtype=torch.float16,
)

print("3. Scheduler setup...", flush=True)
pipe.scheduler = DDIMScheduler.from_config(
    pipe.scheduler.config,
    beta_schedule="linear",
    clip_sample=False,
    timestep_spacing="linspace",
    steps_offset=1,
)

print("4. Attaching LoRA via PeftModel...", flush=True)
peft_unet = PeftModel.from_pretrained(pipe.unet, "models/diffusion_rip_swimmer_lora/final_lora")
pipe.unet = peft_unet.merge_and_unload()
print("   LoRA merged into UNet successfully!", flush=True)

print("5. Enabling GPU & memory optimizations...", flush=True)
pipe.to("cuda")
pipe.enable_vae_slicing()

print("6. Running 1 test generation (8 frames, 10 steps)...", flush=True)
out = pipe(
    prompt="aerial drone photograph of a swimmer caught in an ocean rip current, turbulent foam",
    num_frames=8,
    height=384,
    width=384,
    num_inference_steps=10,
    guidance_scale=7.5,
    generator=torch.Generator("cuda").manual_seed(42),
)
print("7. Inference complete! Frames count:", len(out.frames[0]), flush=True)
out.frames[0][0].save("runs/test_frame0.jpg")
print("8. Saved runs/test_frame0.jpg successfully!", flush=True)
