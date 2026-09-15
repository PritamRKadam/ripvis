import shutil
from pathlib import Path

src_base = Path("runs/stage3_video_generation")
dst_dir = Path("/home/cvl_project_ss26_dumitriu/.gemini/antigravity-ide/brain/efd119c3-1ed8-493a-9314-1b21ba055b85")

files_to_copy = [
    src_base / "scenario_01_rip_neck_swimmer" / "scenario_01_rip_neck_swimmer_keyframes.jpg",
    src_base / "scenario_02_treading_near_waves" / "scenario_02_treading_near_waves_keyframes.jpg",
    src_base / "scenario_04_underwater_caustics_drift" / "scenario_04_underwater_caustics_drift_keyframes.jpg",
    src_base / "scenario_01_rip_neck_swimmer" / "scenario_01_rip_neck_swimmer.webp",
    src_base / "scenario_04_underwater_caustics_drift" / "scenario_04_underwater_caustics_drift.webp",
]

for src in files_to_copy:
    if src.exists():
        dst = dst_dir / src.name
        shutil.copy(src, dst)
        print(f"Copied {src.name} -> {dst}")
