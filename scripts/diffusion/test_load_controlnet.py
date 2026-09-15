import torch
from diffusers import ControlNetModel

print("Testing ControlNet download / load...")
try:
    controlnet = ControlNetModel.from_pretrained(
        "lllyasviel/control_v11p_sd15_openpose",
        torch_dtype=torch.float16,
    )
    print("✓ Successfully loaded lllyasviel/control_v11p_sd15_openpose!")
except Exception as e:
    print("Failed to load v11p:", e)
    try:
        controlnet = ControlNetModel.from_pretrained(
            "lllyasviel/sd-controlnet-openpose",
            torch_dtype=torch.float16,
        )
        print("✓ Successfully loaded lllyasviel/sd-controlnet-openpose!")
    except Exception as e2:
        print("Failed to load sd-controlnet-openpose:", e2)
