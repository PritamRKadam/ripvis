from diffusers import AnimateDiffControlNetPipeline

doc = AnimateDiffControlNetPipeline.__call__.__doc__
for line in doc.split("\n"):
    if "conditioning_frames" in line or "controlnet" in line:
        print(line)
