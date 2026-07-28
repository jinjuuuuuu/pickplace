# bc_train.py  (Action Chunking 버전)
# ---------------------------------------------------------------------------
# pick_place_collect.py 로 수집한 bc_dataset.npz 로 Behavior Cloning 학습.
#
# [핵심 변경] 한 스텝짜리 action을 예측하던 방식은 실패했다. 이유:
#   - 모든 데모가 같은 START_POSE에서 시작 -> 첫 동작이 큐브와 무관(open-loop)
#   - 한 스텝 delta는 큐브 위치와 상관 ~0, 하강/파지 같은 왕복동작이 평균에서 상쇄
#   => 정책이 큐브를 무시하고 평균 궤적만 재생하다 멈춤.
#
# 해결: 정책이 '미래 H스텝의 절대 관절 목표 시퀀스(chunk)'를 한 번에 예측한다.
#   - chunk 라벨은 큐브 위치에 따라 달라지므로, 정책이 큐브 관측을 '쓰도록' 강제됨
#   - 절대 목표라 하강/파지가 평균에서 사라지지 않음
#
# Run (일반 Python, Isaac Sim 불필요):  python bc_train.py
# ---------------------------------------------------------------------------

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split

# === 설정 ===================================================================
DATA_PATH   = r"C:\Users\user\Desktop\claude_jetbot\bc_data_v3\bc_dataset.npz"
SAVE_PATH   = r"C:\Users\user\Desktop\claude_jetbot\bc_data_v3\bc_policy.pt"

CHUNK_H     = 60        # 한 번에 예측하는 미래 스텝 수 (배포의 CHUNK_H와 반드시 동일)
EPOCHS      = 150
BATCH_SIZE  = 256
LR          = 1e-3
VAL_RATIO   = 0.1
# =============================================================================


# ============================================================================
# 1. 데이터 로드 (per-step 절대 관절 목표 + 에피소드 경계)
# ============================================================================
print("[bc_train] 데이터 로드 중...")
data    = np.load(DATA_PATH)
obs     = data["obs"].astype(np.float32)        # (N, 27)
acts    = data["actions"].astype(np.float32)    # (N, 9) per-step ABSOLUTE joint targets

if "episode_starts" in data.files:
    ep_starts = np.array(data["episode_starts"], dtype=np.int64)
else:
    # 경계 정보가 없으면 전체를 하나의 에피소드로 간주 (권장 X)
    print("[bc_train] ⚠ episode_starts 없음 -> 단일 에피소드로 처리")
    ep_starts = np.array([0], dtype=np.int64)

start_pose = data["start_pose"] if "start_pose" in data.files else None

N = len(obs)
OBS_DIM    = obs.shape[1]      # 27
JOINT_DIM  = acts.shape[1]     # 9
ACTION_DIM = CHUNK_H * JOINT_DIM
ep_bounds  = np.append(ep_starts, N)   # [s0, s1, ..., N]

print(f"[bc_train] obs {obs.shape} | per-step act {acts.shape} | 에피소드 {len(ep_starts)}개")
print(f"[bc_train] chunk H={CHUNK_H} -> action_dim={ACTION_DIM}")
if start_pose is not None:
    print(f"[bc_train] 시작 자세: {np.round(start_pose,3)}")


# ============================================================================
# 2. 정규화
#    obs: 27차원 표준화.  action: 관절(9차원) 단위 통계를 H개로 타일링.
# ============================================================================
print("\n[bc_train] 정규화 통계 계산...")
obs_mean = obs.mean(0)
obs_std  = obs.std(0) + 1e-6

# 관절 단위 통계 (절대 목표값 기준). chunk의 모든 H 스텝에 동일 적용.
act_mean = acts.mean(0)              # (9,)
act_std  = acts.std(0) + 1e-6        # (9,)

print(f"[bc_train] obs   정규화 후 min/max: "
      f"{((obs-obs_mean)/obs_std).min():.3f} / {((obs-obs_mean)/obs_std).max():.3f}")

norm_stats_path = SAVE_PATH.replace(".pt", "_norm_stats.pt")
_stats = {
    "obs_mean": torch.tensor(obs_mean, dtype=torch.float32),
    "obs_std":  torch.tensor(obs_std,  dtype=torch.float32),
    "act_mean": torch.tensor(act_mean, dtype=torch.float32),   # (9,)
    "act_std":  torch.tensor(act_std,  dtype=torch.float32),   # (9,)
    "chunk_h":  int(CHUNK_H),
    "joint_dim": int(JOINT_DIM),
}
if start_pose is not None:
    _stats["start_pose"] = torch.tensor(np.array(start_pose), dtype=torch.float32)
torch.save(_stats, norm_stats_path)
print(f"[bc_train] 정규화 통계 저장: {norm_stats_path}")


# ============================================================================
# 3. Chunk Dataset
#    각 프레임 t에 대해 라벨 = [t, t+1, ..., t+H-1]의 절대 관절 목표 (에피소드 내).
#    에피소드 끝을 넘으면 마지막 자세로 패딩(hold).
# ============================================================================
class ChunkDataset(Dataset):
    def __init__(self, obs, acts, ep_bounds, H, obs_mean, obs_std, act_mean, act_std):
        self.obs  = obs
        self.acts = acts
        self.H    = H
        self.om, self.os = obs_mean, obs_std
        self.am, self.as_ = act_mean, act_std
        # 프레임별 에피소드 끝 인덱스(exclusive) 매핑
        self.ep_end = np.empty(len(obs), dtype=np.int64)
        for i in range(len(ep_bounds) - 1):
            s, e = ep_bounds[i], ep_bounds[i + 1]
            self.ep_end[s:e] = e

    def __len__(self):
        return len(self.obs)

    def __getitem__(self, t):
        H, e = self.H, self.ep_end[t]
        idx = np.arange(t, t + H)
        idx = np.minimum(idx, e - 1)          # 에피소드 끝 넘으면 마지막으로 패딩
        chunk = self.acts[idx]                 # (H, 9) 절대 목표
        chunk = (chunk - self.am) / self.as_   # 정규화 (브로드캐스트)
        o = (self.obs[t] - self.om) / self.os
        return (torch.from_numpy(o.astype(np.float32)),
                torch.from_numpy(chunk.reshape(-1).astype(np.float32)))

dataset = ChunkDataset(obs, acts, ep_bounds, CHUNK_H,
                       obs_mean, obs_std, act_mean, act_std)

val_size   = int(len(dataset) * VAL_RATIO)
train_size = len(dataset) - val_size
train_ds, val_ds = random_split(dataset, [train_size, val_size])
train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,  num_workers=0)
val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
print(f"[bc_train] 학습 {train_size} | 검증 {val_size}")


# ============================================================================
# 4. 신경망 (chunk 예측: obs -> H*9)
# ============================================================================
class BCChunkPolicy(nn.Module):
    def __init__(self, obs_dim, action_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, 512), nn.ReLU(),
            nn.Linear(512, 512),     nn.ReLU(),
            nn.Linear(512, 256),     nn.ReLU(),
            nn.Linear(256, action_dim),
        )
    def forward(self, obs):
        return self.net(obs)

device = "cuda" if torch.cuda.is_available() else "cpu"
policy    = BCChunkPolicy(OBS_DIM, ACTION_DIM).to(device)
optimizer = torch.optim.Adam(policy.parameters(), lr=LR)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.5)
loss_fn   = nn.MSELoss()
print(f"[bc_train] device={device} | params={sum(p.numel() for p in policy.parameters()):,}")


# ============================================================================
# 5. 학습 루프
# ============================================================================
print(f"\n[bc_train] 학습 시작 ({EPOCHS} epochs)\n")
best_val = float("inf")
for epoch in range(1, EPOCHS + 1):
    policy.train(); tr = []
    for ob, ch in train_loader:
        ob, ch = ob.to(device), ch.to(device)
        pred = policy(ob)
        loss = loss_fn(pred, ch)
        optimizer.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
        optimizer.step(); tr.append(loss.item())
    scheduler.step()

    policy.eval(); va = []
    with torch.no_grad():
        for ob, ch in val_loader:
            ob, ch = ob.to(device), ch.to(device)
            va.append(loss_fn(policy(ob), ch).item())
    tl, vl = float(np.mean(tr)), float(np.mean(va))
    if epoch % 10 == 0 or epoch == 1:
        print(f"Epoch {epoch:3d}/{EPOCHS} | train {tl:.6f} | val {vl:.6f}")

    if vl < best_val:
        best_val = vl
        torch.save({
            "epoch": int(epoch),
            "policy": policy.state_dict(),
            "obs_dim": int(OBS_DIM),
            "action_dim": int(ACTION_DIM),
            "chunk_h": int(CHUNK_H),
            "joint_dim": int(JOINT_DIM),
            "val_loss": float(vl),
        }, SAVE_PATH)

print(f"\n[bc_train] 완료! best val loss: {best_val:.6f}")
print(f"[bc_train] 저장: {SAVE_PATH}")
