# bc_train_vision_smallcnn_tensorboard.py
# ---------------------------------------------------------------------------
# 기존 순정 SmallCNN + LeRobot 비디오 디코딩 + TensorBoard 그래프
# ---------------------------------------------------------------------------
import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
import torchvision.transforms.functional as TF  # PyTorch 이미지 리사이즈용
import torchvision.models as models
from tqdm import tqdm

# LeRobot의 강력한 비디오 디코더 데이터셋 임포트
try:
    from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
except ImportError:
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

# === 설정 ===================================================================
REPO_ID     = "jamongsteak/pickplace_vision"  
SAVE_PATH   = "bc_policy_vision_smallcnn.pt"

CHUNK_H     = 60          
PROPRIO_DIM = 18          
IMG_SIZE    = 84          
EPOCHS      = 100
BATCH_SIZE  = 64
LR          = 1e-3
VAL_RATIO   = 0.1
# =============================================================================

device = "cuda" if torch.cuda.is_available() else "cpu"

# ============================================================================
# 1. 데이터 로드 및 정규화 통계 계산 (LeRobotDataset 사용)
# ============================================================================
print(f"[vis_train_smallcnn] 데이터셋 '{REPO_ID}' 로드 중... (비디오 디코딩 준비)")
# 일반 load_dataset 대신 LeRobotDataset을 사용하여 mp4 비디오를 쉽게 읽게 합니다.
lerobot_dataset = LeRobotDataset(REPO_ID)
hf_dataset = lerobot_dataset.hf_dataset  # 통계용 원본 데이터 접근
N = len(lerobot_dataset)

# 통계 계산 (np.asarray로 감싸서 계산 오류 방지)
states = np.asarray(hf_dataset["observation.state"])
actions = np.asarray(hf_dataset["action"])
ep_indices = np.asarray(hf_dataset["episode_index"])

pro_mean, pro_std = states.mean(0), states.std(0) + 1e-6
act_mean, act_std = actions.mean(0), actions.std(0) + 1e-6
JOINT_DIM = actions.shape[1]
ACTION_DIM = CHUNK_H * JOINT_DIM

ep_end = np.empty(N, dtype=np.int64)
unique_eps, counts = np.unique(ep_indices, return_counts=True)
end_idx = 0
for count in counts:
    ep_end[end_idx : end_idx+count] = end_idx + count
    end_idx += count

print(f"[vis_train_smallcnn] 프레임 {N} | 에피소드 {len(unique_eps)} | action_dim={ACTION_DIM}")

stats_path = SAVE_PATH.replace(".pt", "_norm_stats.pt")
torch.save({
    "proprio_mean": torch.tensor(pro_mean, dtype=torch.float32),
    "proprio_std":  torch.tensor(pro_std,  dtype=torch.float32),
    "act_mean":     torch.tensor(act_mean, dtype=torch.float32),
    "act_std":      torch.tensor(act_std,  dtype=torch.float32),
    "chunk_h": int(CHUNK_H), "joint_dim": int(JOINT_DIM),
    "proprio_dim": int(PROPRIO_DIM), "img_size": int(IMG_SIZE),
}, stats_path)

# ============================================================================
# 2. Chunk Dataset (비디오 프레임 추출 특화)
# ============================================================================
class HFVisChunkDataset(Dataset):
    def __init__(self, idxs):
        self.idxs = idxs
    def __len__(self): 
        return len(self.idxs)
    def __getitem__(self, i):
        t = self.idxs[i]; e = ep_end[t]
        idx = np.minimum(np.arange(t, t + CHUNK_H), e - 1)
        chunk = (actions[idx] - act_mean) / act_std
        
        # 💡 매직 포인트: lerobot_dataset[t]를 호출하면 백그라운드에서 mp4 비디오를 
        # 디코딩하여 이미지를 파이토치 Tensor(C, H, W, 0.0~1.0) 형태로 바로 뱉어줍니다!
        frame = lerobot_dataset[int(t)]
        w_img = frame["observation.images.wrist"]
        o_img = frame["observation.images.over"]
        pro_raw = np.asarray(frame["observation.state"])
        
        # PIL 변환 없이 파이토치 자체 기능으로 초고속 리사이즈 (120x160 -> 84x84)
        w = TF.resize(w_img, [IMG_SIZE, IMG_SIZE], antialias=True)
        o = TF.resize(o_img, [IMG_SIZE, IMG_SIZE], antialias=True)
        pro = (pro_raw - pro_mean) / pro_std
        
        return (w, o,
                torch.from_numpy(pro.astype(np.float32)),
                torch.from_numpy(chunk.reshape(-1).astype(np.float32)))

all_idx = np.arange(N)
val_n = int(N * VAL_RATIO)
rng = np.random.default_rng(0); rng.shuffle(all_idx)
val_idx, train_idx = all_idx[:val_n], all_idx[val_n:]

train_loader = DataLoader(HFVisChunkDataset(train_idx), batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
val_loader   = DataLoader(HFVisChunkDataset(val_idx),   batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

# ============================================================================
# 3. 신경망 — 기존 순정 SmallCNN 유지 파트
# ============================================================================
class SpatialSoftmax(nn.Module):
    def forward(self, x):
        B, C, H, W = x.shape
        a = torch.softmax(x.reshape(B, C, H * W), dim=-1)
        ys, xs = torch.meshgrid(torch.linspace(-1, 1, H, device=x.device),
                                torch.linspace(-1, 1, W, device=x.device), indexing="ij")
        ex = (a * xs.reshape(-1)).sum(-1)
        ey = (a * ys.reshape(-1)).sum(-1)
        return torch.cat([ex, ey], dim=1)

class SmallCNN(nn.Module):
    def __init__(self, out_dim=128):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3, 32, 5, stride=2, padding=2), nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(64, 32, 3, stride=1, padding=1), nn.ReLU(),
        )
        self.sm = SpatialSoftmax()
        self.fc = nn.Linear(64, out_dim)
    def forward(self, x):
        return torch.relu(self.fc(self.sm(self.conv(x))))

class VisionBCPolicy(nn.Module):
    def __init__(self, proprio_dim, action_dim, feat=128):
        super().__init__()
        self.cnn_wrist = SmallCNN(feat)
        self.cnn_over  = SmallCNN(feat)
        self.head = nn.Sequential(
            nn.Linear(feat*2 + proprio_dim, 512), nn.ReLU(),
            nn.Linear(512, 512), nn.ReLU(),
            nn.Linear(512, 256), nn.ReLU(),
            nn.Linear(256, action_dim),
        )
    def forward(self, wrist_img, over_img, proprio):
        fw = self.cnn_wrist(wrist_img); fo = self.cnn_over(over_img)
        return self.head(torch.cat([fw, fo, proprio], dim=1))

policy    = VisionBCPolicy(PROPRIO_DIM, ACTION_DIM).to(device)
optimizer = torch.optim.Adam(policy.parameters(), lr=LR)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=40, gamma=0.5)
loss_fn   = nn.MSELoss()

writer = SummaryWriter("runs/bc_vision_smallcnn")
print(f"[vis_train_smallcnn] device={device} | params={sum(p.numel() for p in policy.parameters()):,}")

# ============================================================================
# 4. 학습 루프
# ============================================================================
print(f"\n[vis_train_smallcnn] 학습 시작 ({EPOCHS} epochs)\n")
best_val = float("inf")
for epoch in range(1, EPOCHS + 1):
    policy.train(); tr = []
    for w, o, pro, ch in tqdm(train_loader, desc=f"Epoch {epoch} [Train]"):
        w, o, pro, ch = w.to(device), o.to(device), pro.to(device), ch.to(device)   
        pred = policy(w, o, pro)
        loss = loss_fn(pred, ch)
        optimizer.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
        optimizer.step(); tr.append(loss.item())
    scheduler.step()

    policy.eval(); va = []
    with torch.no_grad():
        for w, o, pro, ch in val_loader:
            w, o, pro, ch = w.to(device), o.to(device), pro.to(device), ch.to(device)
            va.append(loss_fn(policy(w, o, pro), ch).item())
            
    tl, vl = float(np.mean(tr)), float(np.mean(va))
    if epoch % 5 == 0 or epoch == 1:
        print(f"Epoch {epoch:3d}/{EPOCHS} | train {tl:.6f} | val {vl:.6f}")

    writer.add_scalar("Loss/train", tl, epoch)
    writer.add_scalar("Loss/val", vl, epoch)

    if vl < best_val:
        best_val = vl
        torch.save({
            "epoch": int(epoch), "policy": policy.state_dict(),
            "proprio_dim": int(PROPRIO_DIM), "action_dim": int(ACTION_DIM),
            "chunk_h": int(CHUNK_H), "joint_dim": int(JOINT_DIM),
            "img_size": int(IMG_SIZE), "val_loss": float(vl),
        }, SAVE_PATH)

print(f"\n[vis_train_smallcnn] 완료! best val loss: {best_val:.6f}")
writer.close()