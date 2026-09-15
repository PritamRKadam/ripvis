import os
import glob
import cv2

mixkit_dir = "/home/cvl_project_ss26_dumitriu/Desktop/Videos"
drone_dir = "/home/cvl_project_ss26_dumitriu/Desktop/Videos/MOBDrone_videos/video_split_fullhd"

mixkit_videos = sorted([f for f in os.listdir(mixkit_dir) if f.endswith(".mp4")])
drone_videos = sorted([f for f in os.listdir(drone_dir) if f.endswith(".mp4")])

print(f"Mixkit/Envato videos in Desktop/Videos: {len(mixkit_videos)}")
print(f"Drone videos in Desktop/Videos/MOBDrone_videos/video_split_fullhd: {len(drone_videos)}")

# Check first 3 drone videos
for f in drone_videos[:3]:
    p = os.path.join(drone_dir, f)
    cap = cv2.VideoCapture(p)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    cnt = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    print(f"Drone sample {f}: {w}x{h} @ {fps:.1f}fps, {cnt} frames ({cnt/fps:.1f}s)")
