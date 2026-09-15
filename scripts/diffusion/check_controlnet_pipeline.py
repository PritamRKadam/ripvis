import torch
from diffusers import ControlNetModel, AnimateDiffControlNetPipeline, MotionAdapter
import inspect

print("Checking ControlNet and AnimateDiffControlNetPipeline...")
sig = inspect.signature(AnimateDiffControlNetPipeline.__init__)
print("AnimateDiffControlNetPipeline.__init__ params:")
for p in sig.parameters.values():
    print(f"  {p.name}: {p.default}")
