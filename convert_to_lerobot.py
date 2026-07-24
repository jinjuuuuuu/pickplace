#!/usr/bin/env python3
# convert_to_lerobot.py  (비전 버전: 이미지 + 상태)
# ---------------------------------------------------------------------------
# WSL2의 conda 'lerobot' 환경에서 실행:
#   conda activate lerobot
#   python /mnt/c/Users/user/Desktop/claude_jetbot/convert_to_lerobot.py
#
# 하는 일: bc_data_v6의 episode_*.npz (obs[T,27], actions[T,9],
#          images_wrist[T,H,W,3], images_over[T,H,W,3]) 를
#          LeRobotDataset(이미지+상태 입력) 형식으로 변환한다.
# ---------------------------------------------------------------------------
import os, glob, numpy as np
from lerobot.datasets.lerobot_dataset import LeRobotDataset

# ---- 설정 ----
# 방금 수집한 v6 데이터 폴더로 경로 업데이트!
SRC     = "/mnt/c/Users/user/Desktop/claude_jetbot/bc_data_v5"
REPO_ID = "jamongsteak/pickplace_vision_v5"   
FPS     = 30
TASK    = "pick up the cube and place it on the target"
USE_VIDEOS = True                                 # 이미지를 mp4로 인코딩(용량↓). 문제 시 False

# ---- LeRobot import (버전별 경로 방어) ----
try:
    from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
except ImportError:
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

files = sorted(glob.glob(os.path.join(SRC, "episode_*.npz")))
assert files, f"에피소드 파일이 없습니다: {SRC}"

d0 = np.load(files[0])
assert "images_wrist" in d0.files and "images_over" in d0.files, (
    f"이미지가 없는 데이터입니다. RECORD_IMAGES=True 로 다시 수집하세요. keys={d0.files}")

OBS_DIM = int(d0["obs"].shape[1])
ACT_DIM = int(d0["actions"].shape[1])

# 🔥 핵심 수정: 상태는 관절 위치(joint_pos 9차원)만 가져오도록 수정!
# obs 배열의 구조: [joint_pos(9), joint_vel(9), cube_rel(3), target_rel(3), cube_pos(3)]
PROPRIO_DIM = 9 

H_W, W_W = int(d0["images_wrist"].shape[1]), int(d0["images_wrist"].shape[2])
H_O, W_O = int(d0["images_over"].shape[1]),  int(d0["images_over"].shape[2])
print(f"[convert] 에피소드 {len(files)}개 | obs={OBS_DIM}→state {PROPRIO_DIM} act={ACT_DIM} "
      f"| wrist={H_W}x{W_W} over={H_O}x{W_O}")

features = {
    "observation.images.wrist": {"dtype": "video" if USE_VIDEOS else "image",
                                  "shape": (H_W, W_W, 3),
                                  "names": ["height", "width", "channels"]},
    "observation.images.over":  {"dtype": "video" if USE_VIDEOS else "image",
                                  "shape": (H_O, W_O, 3),
                                  "names": ["height", "width", "channels"]},
    "observation.state": {"dtype": "float32", "shape": (PROPRIO_DIM,), "names": None},
    "action":            {"dtype": "float32", "shape": (ACT_DIM,), "names": None},
}

dataset = LeRobotDataset.create(
    repo_id=REPO_ID,
    fps=FPS,
    features=features,
    robot_type="franka",
    use_videos=USE_VIDEOS,
)

# ---- add_frame: task 인자 위치가 버전마다 달라서 첫 프레임에서 자동 판별 ----
_add_mode = {"v": None}
def add_frame(frame, task):
    m = _add_mode["v"]
    if m == "kw":     return dataset.add_frame(frame, task=task)
    if m == "indict": return dataset.add_frame({**frame, "task": task})
    if m == "plain":  return dataset.add_frame(frame)
    try:
        dataset.add_frame(frame, task=task); _add_mode["v"] = "kw"; return
    except TypeError:
        pass
    try:
        dataset.add_frame({**frame, "task": task}); _add_mode["v"] = "indict"; return
    except TypeError:
        pass
    dataset.add_frame(frame); _add_mode["v"] = "plain"

def save_ep(task):
    try:
        dataset.save_episode()
    except TypeError:
        dataset.save_episode(task=task)

def to_img(a):
    """HxWx3 uint8 로 정리 (LeRobot 이미지 입력 형식)."""
    a = np.asarray(a)
    if a.ndim == 3 and a.shape[2] >= 3:
        a = a[:, :, :3]
    if a.dtype != np.uint8:
        a = np.clip(a, 0, 255).astype(np.uint8)
    return np.ascontiguousarray(a)

total = 0
for f in files:
    d = np.load(f)
    obs = np.asarray(d["obs"],     dtype=np.float32)
    act = np.asarray(d["actions"], dtype=np.float32)
    iw  = d["images_wrist"]
    io  = d["images_over"]
    T = min(len(obs), len(act), len(iw), len(io))
    for t in range(T):
        add_frame({
            "observation.images.wrist": to_img(iw[t]),
            "observation.images.over":  to_img(io[t]),
            "observation.state":        obs[t][:PROPRIO_DIM],
            "action":                   act[t],
        }, TASK)
    save_ep(TASK)
    total += T
    print(f"[convert]  {os.path.basename(f)}: {T} frames")

# ---- 마무리 (v3=finalize / 구버전=consolidate) ----
if hasattr(dataset, "finalize"):
    dataset.finalize()
elif hasattr(dataset, "consolidate"):
    dataset.consolidate()

dataset.push_to_hub(REPO_ID)

print(f"[convert] 완료! 총 {total} 프레임, {len(files)} 에피소드")