import os
import glob
import cv2
import numpy as np

drone_dir = "/home/cvl_project_ss26_dumitriu/Desktop/Videos/MOBDrone_videos/video_split_fullhd"
files = sorted(glob.glob(os.path.join(drone_dir, "*.mp4")))
print(f"Total MOBDrone mp4s found: {len(files)}")

# Inspect sample filenames and extract attributes
altitudes = set()
for f in files:
    name = os.path.basename(f)
    # DJI_0804_0001_30m_1.mp4
    parts = name.replace(".mp4", "").split("_")
    for p in parts:
        if p.endswith("m") and p[:-1].isdigit():
            altitudes.add(int(p[:-1]))

print(f"Detected altitudes in MOBDrone: {sorted(list(altitudes))} meters")

# Sample a few files to check dimensions and duration
for f in files[:3]:
    name = os.path.basename(f)
    cap = cv2.VideoCapture(f)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    cnt = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    print(f"{name}: {w}x{h} @ {fps:.1f} fps, {cnt} frames ({cnt/fps:.1f}s)")
