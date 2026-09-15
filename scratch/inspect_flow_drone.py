import glob
import os
import cv2
import numpy as np

drone_dir = "/home/cvl_project_ss26_dumitriu/Desktop/Videos/MOBDrone_videos/video_split_fullhd"
files = sorted(glob.glob(os.path.join(drone_dir, "*.mp4")))

# Sample 5 files across different altitudes
sample_files = [files[0], files[15], files[30], files[45], files[60]]

for f in sample_files:
    cap = cv2.VideoCapture(f)
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # Read frame 0 and frame 15 to compute Farneback optical flow
    ret, f1 = cap.read()
    for _ in range(14):
        cap.read()
    ret, f2 = cap.read()
    cap.release()
    
    if ret and f1 is not None and f2 is not None:
        g1 = cv2.cvtColor(cv2.resize(f1, (320, 180)), cv2.COLOR_BGR2GRAY)
        g2 = cv2.cvtColor(cv2.resize(f2, (320, 180)), cv2.COLOR_BGR2GRAY)
        flow = cv2.calcOpticalFlowFarneback(g1, g2, None, 0.5, 3, 15, 3, 5, 1.2, 0)
        mag, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])
        mean_flow = float(np.mean(mag))
        
        # Check average brightness and blue/green ratio
        mean_b = float(np.mean(f1[:, :, 0]))
        mean_g = float(np.mean(f1[:, :, 1]))
        mean_r = float(np.mean(f1[:, :, 2]))
        
        print(f"{os.path.basename(f)}: flow_mean={mean_flow:.2f}, RGB=({mean_r:.1f},{mean_g:.1f},{mean_b:.1f}), frames={n_frames}")
