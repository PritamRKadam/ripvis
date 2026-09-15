import inspect
from diffusers import AnimateDiffVideoToVideoPipeline

sig = inspect.signature(AnimateDiffVideoToVideoPipeline.__call__)
print("AnimateDiffVideoToVideoPipeline parameters:")
for p in sig.parameters.values():
    print(f"  {p.name}: {p.default}")
