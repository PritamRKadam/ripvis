import sys

print("Python version:", sys.version)

for mod in ["torch", "torchvision", "ultralytics", "cv2", "PIL", "scipy", "sklearn"]:
    try:
        m = __import__(mod)
        print(f"  {mod}: available (v{getattr(m, '__version__', 'unknown')})")
    except ImportError:
        print(f"  {mod}: NOT available")
