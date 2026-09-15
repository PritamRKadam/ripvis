import inspect
from diffusers import AnimateDiffControlNetPipeline

sig = inspect.signature(AnimateDiffControlNetPipeline.__call__)
print("AnimateDiffControlNetPipeline.__call__ params:")
for p in sig.parameters.values():
    print(f"  {p.name}: {p.default}")
