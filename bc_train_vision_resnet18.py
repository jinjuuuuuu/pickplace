# bc_train_vision.py
# ---------------------------------------------------------------------------
# ResNet18 백본 + [과거 3프레임 스태킹(기억력)] + TQDM + TensorBoard
# ---------------------------------------------------------------------------
import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
import torchvision.transforms.functional as TF
import torchvision.models as models
from tqdm import tqdm

try:
    from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
except ImportError:
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

# === 설정 ===================================================================
REPO_ID     = "jamongsteak/pickplace_vision_v5"   # convert_to_lerobot.py의 REPO_ID와 일치시킬 것
SAVE_PATH   = "bc_policy_vision_resnet_stacked.pt"  # 헷갈리지 않게 이름 변경

CHUNK_H     = 60
PROPRIO_DIM = 9           # observation.state = joint_pos(9)만. convert_to_lerobot.py의 PROPRIO_DIM과 일치
IMG_SIZE    = 84
EPOCHS      = 200
BATCH_SIZE  = 64
LR          = 1e-3
VAL_RATIO   = 0.1

OBS_SEQ     = 3  # 🔥 [핵심] 현재를 포함해 과거 몇 장의 프레임을 볼 것인가? (t-2, t-1, t)
# =============================================================================

device = "cuda" if torch.cuda.is_available() else "cpu"

# ============================================================================
# 1. 데이터 로드 및 통계 계산 
# ============================================================================
print(f"[vis_train_resnet_stack] 데이터셋 '{REPO_ID}' 로드 중...")
lerobot_dataset = LeRobotDataset(REPO_ID)
hf_dataset = lerobot_dataset.hf_dataset  
N = len(lerobot_dataset)

states = np.asarray(hf_dataset["observation.state"])
actions = np.asarray(hf_dataset["action"])
ep_indices = np.asarray(hf_dataset["episode_index"])

# 🔒 안전장치: 데이터셋의 실제 상태 차원과 PROPRIO_DIM이 다르면 즉시 중단
# (조용한 차원 불일치로 학습이 엉뚱하게 도는 것을 방지)
assert states.shape[1] == PROPRIO_DIM, (
    f"PROPRIO_DIM({PROPRIO_DIM})와 데이터셋 observation.state 차원"
    f"({states.shape[1]})이 다릅니다. 설정을 데이터에 맞추세요.")

pro_mean, pro_std = states.mean(0), states.std(0) + 1e-6
act_mean, act_std = actions.mean(0), actions.std(0) + 1e-6
JOINT_DIM = actions.shape[1]
ACTION_DIM = CHUNK_H * JOINT_DIM

# 🔥 에피소드의 시작(ep_start)과 끝(ep_end)을 모두 기록합니다.
# (과거 프레임을 찾을 때 이전 에피소드 사진을 훔쳐보지 않게 막기 위함)
ep_start = np.empty(N, dtype=np.int64)
ep_end = np.empty(N, dtype=np.int64)
unique_eps, counts = np.unique(ep_indices, return_counts=True)
idx_counter = 0
for count in counts:
    ep_start[idx_counter : idx_counter+count] = idx_counter
    ep_end[idx_counter : idx_counter+count] = idx_counter + count
    idx_counter += count

# ============================================================================
# 2. Chunk Dataset (프레임 스태킹 로직 추가)
# ============================================================================
class HFVisChunkDataset(Dataset):
    def __init__(self, idxs):
        self.idxs = idxs
    def __len__(self): 
        return len(self.idxs)
    def __getitem__(self, i):
        t = self.idxs[i]
        e = ep_end[t]
        s = ep_start[t]
        
        idx = np.minimum(np.arange(t, t + CHUNK_H), e - 1)
        chunk = (actions[idx] - act_mean) / act_std
        
        # 🔥 t-2, t-1, t 프레임의 인덱스를 구합니다. (에피소드 시작점 s 밖으로 나가지 않게 방어)
        obs_idx = np.clip(np.arange(t - OBS_SEQ + 1, t + 1), s, e - 1)
        
        w_imgs, o_imgs, pros = [], [], []
        for curr_t in obs_idx:
            frame = lerobot_dataset[int(curr_t)]
            w = TF.resize(frame["observation.images.wrist"], [IMG_SIZE, IMG_SIZE], antialias=True)
            o = TF.resize(frame["observation.images.over"], [IMG_SIZE, IMG_SIZE], antialias=True)
            p = (np.asarray(frame["observation.state"]) - pro_mean) / pro_std
            w_imgs.append(w)
            o_imgs.append(o)
            pros.append(p)
            
        # 🔥 사진 3장을 채널 방향(dim=0)으로 겹쳐서 9채널(3*3)로 만듭니다.
        w_stacked = torch.cat(w_imgs, dim=0)
        o_stacked = torch.cat(o_imgs, dim=0)
        pro_stacked = torch.tensor(np.concatenate(pros), dtype=torch.float32)
        
        return (w_stacked, o_stacked, pro_stacked, torch.from_numpy(chunk.reshape(-1).astype(np.float32)))

# ============================================================================
# 3. 신경망 (9채널을 받도록 ResNet 수술)
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

class ResNet18Backbone(nn.Module):
    # 🔥 in_channels가 기본 3에서 9(3프레임)로 바뀝니다.
    def __init__(self, in_channels=3 * OBS_SEQ, out_dim=128):
        super().__init__()
        resnet = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        # 🔥 입력 채널을 9개로 수술합니다!
        resnet.conv1 = nn.Conv2d(in_channels, 64, kernel_size=3, stride=1, padding=1, bias=False)
        resnet.maxpool = nn.Identity()
        
        self.backbone = nn.Sequential(
            resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool,
            resnet.layer1, resnet.layer2, resnet.layer3, resnet.layer4,
        )
        self.sm = SpatialSoftmax()
        self.fc = nn.Linear(512 * 2, out_dim)

    def forward(self, x):
        return torch.relu(self.fc(self.sm(self.backbone(x))))

class VisionBCPolicy(nn.Module):
    def __init__(self, proprio_dim, action_dim, obs_seq=OBS_SEQ, feat=128):
        super().__init__()
        self.cnn_wrist = ResNet18Backbone(in_channels=3 * obs_seq, out_dim=feat)
        self.cnn_over  = ResNet18Backbone(in_channels=3 * obs_seq, out_dim=feat)
        
        # 🔥 뇌로 들어가는 정보도 3배(proprio_dim * obs_seq) 늘어났으므로 크기를 맞춰줍니다.
        self.head = nn.Sequential(
            nn.Linear(feat*2 + proprio_dim * obs_seq, 512), nn.ReLU(),
            nn.Linear(512, 512), nn.ReLU(),
            nn.Linear(512, 256), nn.ReLU(),
            nn.Linear(256, action_dim),
        )
    def forward(self, wrist_img, over_img, proprio):
        fw = self.cnn_wrist(wrist_img); fo = self.cnn_over(over_img)
        return self.head(torch.cat([fw, fo, proprio], dim=1))

# ============================================================================
# 4. 학습 루프
# ============================================================================
if __name__ == '__main__':
    stats_path = SAVE_PATH.replace(".pt", "_norm_stats.pt")
    torch.save({
        "proprio_mean": torch.tensor(pro_mean, dtype=torch.float32),
        "proprio_std":  torch.tensor(pro_std,  dtype=torch.float32),
        "act_mean":     torch.tensor(act_mean, dtype=torch.float32),
        "act_std":      torch.tensor(act_std,  dtype=torch.float32),
        "chunk_h": int(CHUNK_H), "joint_dim": int(JOINT_DIM),
        "proprio_dim": int(PROPRIO_DIM), "img_size": int(IMG_SIZE),
        "obs_seq": int(OBS_SEQ)  # 🔥 나중에 배포할 때 쓰기 위해 저장해둡니다.
    }, stats_path)

    all_idx = np.arange(N)
    val_n = int(N * VAL_RATIO)
    rng = np.random.default_rng(0); rng.shuffle(all_idx)
    val_idx, train_idx = all_idx[:val_n], all_idx[val_n:]

    train_loader = DataLoader(HFVisChunkDataset(train_idx), batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
    val_loader   = DataLoader(HFVisChunkDataset(val_idx),   batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

    policy    = VisionBCPolicy(PROPRIO_DIM, ACTION_DIM).to(device)
    optimizer = torch.optim.Adam(policy.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=40, gamma=0.5)
    loss_fn   = nn.MSELoss()

    writer = SummaryWriter("runs/bc_vision_resnet_stacked")
    print(f"[vis_train_resnet_stack] 학습 시작! 파라미터 수={sum(p.numel() for p in policy.parameters()):,}\n")
    
    best_val = float("inf")
    for epoch in range(1, EPOCHS + 1):
        policy.train(); tr = []
        for w, o, pro, ch in tqdm(train_loader, desc=f"Epoch {epoch:3d} [Train]"):
            w, o, pro, ch = w.to(device), o.to(device), pro.to(device), ch.to(device)
            pred = policy(w, o, pro)
            loss = loss_fn(pred, ch)
            optimizer.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
            optimizer.step(); tr.append(loss.item())
        scheduler.step()

        policy.eval(); va = []
        with torch.no_grad():
            for w, o, pro, ch in tqdm(val_loader, desc=f"Epoch {epoch:3d} [Val  ]"):
                w, o, pro, ch = w.to(device), o.to(device), pro.to(device), ch.to(device)
                va.append(loss_fn(policy(w, o, pro), ch).item())
                
        tl, vl = float(np.mean(tr)), float(np.mean(va))
        print(f"➜ 결과 | train_loss: {tl:.6f} | val_loss: {vl:.6f}\n")

        writer.add_scalar("Loss/train", tl, epoch)
        writer.add_scalar("Loss/val", vl, epoch)

        if vl < best_val:
            best_val = vl
            torch.save({
                "epoch": int(epoch), "policy": policy.state_dict(),
                "val_loss": float(vl),
            }, SAVE_PATH)

    writer.close()