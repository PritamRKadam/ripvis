import inspect
import diffusers

print("Available AnimateDiff pipelines:")
for attr in dir(diffusers):
    if "animatediff" in attr.lower():
        print(" -", attr)
