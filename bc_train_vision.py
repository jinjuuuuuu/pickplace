# bc_train_vision.py  (v11_s3 대응판 — 원본은 bc_train_vision.py.orig.bak)
# ---------------------------------------------------------------------------
# 기존 순정 SmallCNN BC 정책을 jamongsteak/pickplace_vision_v11_s3로 학습한다.
# 신경망(SpatialSoftmax + SmallCNN 2대 + MLP 헤드)은 원본 그대로다.
#
# 원본에서 바뀐 것과 그 이유:
#   1) 데이터를 LeRobotDataset이 아니라 prepare_bc_data.py가 구운 npy 캐시에서 읽는다.
#      v11_s3는 LeRobot v3.0 포맷이라 이 PC의 lerobot 0.4.4로는 열리지 않고,
#      프레임마다 mp4를 랜덤 디코딩하면 100에폭이 며칠 걸린다.
#   2) PROPRIO_DIM 18 -> 9. v11 데이터셋의 observation.state는 관절위치 9개뿐이다
#      (수집 스크립트가 obs[:9]만 저장한다). 평가 클라이언트도 jp 9개를 보낸다.
#   3) 검증 분할을 프레임 랜덤 -> 에피소드 단위로. 20Hz에서 이웃 프레임은 거의
#      같은 그림이라, 프레임을 랜덤으로 쪼개면 검증셋이 학습셋에 새어 들어가
#      val loss가 항상 낮게 나온다(일반화 신호가 아니게 된다).
#   4) 에폭마다 체크포인트를 남긴다. 이 프로젝트에는 val loss로 성공률을 예측할
#      근거가 없으므로(README 참고), 결국 여러 체크포인트를 실제로 평가해서 고른다.
#
# 실행 (GPU가 있는 Isaac Sim 번들 파이썬 — torch 2.7+cu128):
#   C:\isaacsim\python.bat -u bc_train_vision.py
#
# 환경변수로 조절: CACHE_DIR EPOCHS BATCH_SIZE LR CHUNK_H VAL_RATIO SAVE_EVERY OUT_DIR
# ---------------------------------------------------------------------------
import csv
import os
import time

import numpy as np
import torch
import torch.nn as nn

# === 설정 ===================================================================
HERE = os.path.dirname(os.path.abspath(__file__))
CACHE_DIR = os.environ.get("CACHE_DIR", os.path.join(HERE, "bc_cache_pickplace_vision_v11_s3_84"))
OUT_DIR = os.environ.get("OUT_DIR", os.path.join(HERE, "bc_runs", "v11_s3_smallcnn"))

# CHUNK_H는 "몇 스텝치 액션을 한 번에 뱉는가"다. v11_s3는 stride 3으로 솎아
# 20Hz이므로 60스텝 = 3초다(원본 60Hz 데이터에서는 60스텝 = 1초였다).
CHUNK_H = int(os.environ.get("CHUNK_H", "60"))
PROPRIO_DIM = int(os.environ.get("PROPRIO_DIM", "9"))
EPOCHS = int(os.environ.get("EPOCHS", "100"))
BATCH_SIZE = int(os.environ.get("BATCH_SIZE", "64"))
LR = float(os.environ.get("LR", "1e-3"))
VAL_RATIO = float(os.environ.get("VAL_RATIO", "0.1"))
SAVE_EVERY = int(os.environ.get("SAVE_EVERY", "10"))
SEED = int(os.environ.get("SEED", "0"))
# =============================================================================

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(SEED)

# ============================================================================
# 1. 캐시 로드 + 정규화 통계
# ============================================================================
meta = np.load(os.path.join(CACHE_DIR, "meta.npz"), allow_pickle=True)
IMG_SIZE = int(meta["img_size"])
FPS = int(meta["fps"])
REPO_ID = str(meta["repo_id"])
states = meta["states"].astype(np.float32)
actions = meta["actions"].astype(np.float32)
ep_index = meta["episode_index"].astype(np.int64)

if states.shape[1] != PROPRIO_DIM:
    raise SystemExit(f"[bc_train] state 차원 불일치: 데이터 {states.shape[1]} vs "
                     f"PROPRIO_DIM {PROPRIO_DIM}. 환경변수로 맞추거나 캐시를 다시 구울 것.")

wrist = np.load(os.path.join(CACHE_DIR, f"wrist_{IMG_SIZE}.npy"), mmap_mode="r")
over = np.load(os.path.join(CACHE_DIR, f"over_{IMG_SIZE}.npy"), mmap_mode="r")

N = len(states)
JOINT_DIM = actions.shape[1]
ACTION_DIM = CHUNK_H * JOINT_DIM
n_eps = int(ep_index.max()) + 1

pro_mean, pro_std = states.mean(0), states.std(0) + 1e-6
act_mean, act_std = actions.mean(0), actions.std(0) + 1e-6

print(f"[bc_train] {REPO_ID} | 프레임 {N} | 에피소드 {n_eps} | {FPS}Hz | img {IMG_SIZE}")
print(f"[bc_train] chunk_h={CHUNK_H} ({CHUNK_H/FPS:.1f}초) joint_dim={JOINT_DIM} "
      f"action_dim={ACTION_DIM} proprio_dim={PROPRIO_DIM}")

# 각 프레임이 속한 에피소드의 끝 인덱스(exclusive). 청크가 에피소드를 넘어가면
# 마지막 프레임으로 클램프한다 (원본 동작 유지).
ep_end = np.zeros(N, dtype=np.int64)
bounds = np.searchsorted(ep_index, np.arange(n_eps + 1))
for e in range(n_eps):
    ep_end[bounds[e]:bounds[e + 1]] = bounds[e + 1]

os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(os.path.join(OUT_DIR, "checkpoints"), exist_ok=True)
norm_stats = {
    "proprio_mean": torch.tensor(pro_mean), "proprio_std": torch.tensor(pro_std),
    "act_mean": torch.tensor(act_mean), "act_std": torch.tensor(act_std),
    "chunk_h": CHUNK_H, "joint_dim": JOINT_DIM, "proprio_dim": PROPRIO_DIM,
    "img_size": IMG_SIZE, "fps": FPS, "repo_id": REPO_ID,
}
torch.save(norm_stats, os.path.join(OUT_DIR, "norm_stats.pt"))

# ============================================================================
# 2. 에피소드 단위 분할 + GPU 상주 텐서
# ============================================================================
rng = np.random.default_rng(SEED)
ep_perm = rng.permutation(n_eps)
n_val_ep = max(1, int(round(n_eps * VAL_RATIO)))
val_eps = set(ep_perm[:n_val_ep].tolist())
is_val = np.isin(ep_index, list(val_eps))
val_idx = np.nonzero(is_val)[0]
train_idx = np.nonzero(~is_val)[0]
print(f"[bc_train] 검증: 에피소드 {sorted(val_eps)} ({len(val_idx)} 프레임) | "
      f"학습 {len(train_idx)} 프레임")

# 청크 타깃 인덱스 (N, CHUNK_H). 미리 만들어 두면 배치마다 gather 한 번이면 된다.
chunk_idx = np.minimum(np.arange(N)[:, None] + np.arange(CHUNK_H)[None, :],
                       (ep_end - 1)[:, None])

act_norm_t = torch.from_numpy((actions - act_mean) / act_std).to(device)
chunk_idx_t = torch.from_numpy(chunk_idx).to(device)
pro_norm_t = torch.from_numpy((states - pro_mean) / pro_std).to(device)

# 이미지는 1GB 남짓이라 VRAM에 통째로 올린다(배치마다 PCIe 전송이 사라진다).
# 안 올라가면 CPU 고정메모리로 떨어진다.
img_bytes = wrist.nbytes + over.nbytes
def _to_device(arr, name):
    t = torch.from_numpy(np.ascontiguousarray(arr))
    if device.type == "cuda":
        free, _ = torch.cuda.mem_get_info()
        if free > img_bytes * 1.3:
            return t.to(device)
    return t.pin_memory() if device.type == "cuda" else t

t0 = time.time()
wrist_t = _to_device(wrist, "wrist")
over_t = _to_device(over, "over")
print(f"[bc_train] 이미지 {img_bytes/1e9:.2f}GB → {wrist_t.device} ({time.time()-t0:.1f}s)")


def make_batch(idx_t):
    """uint8 NHWC -> float NCHW(0~1), 정규화된 proprio, 평탄화된 청크 타깃."""
    img_idx = idx_t if idx_t.device == wrist_t.device else idx_t.to(wrist_t.device)
    w = wrist_t[img_idx].to(device, non_blocking=True).permute(0, 3, 1, 2).float().div_(255.0)
    o = over_t[img_idx].to(device, non_blocking=True).permute(0, 3, 1, 2).float().div_(255.0)
    pro = pro_norm_t[idx_t]
    tgt = act_norm_t[chunk_idx_t[idx_t]].reshape(len(idx_t), -1)
    return w, o, pro, tgt


# ============================================================================
# 3. 신경망 — 원본 순정 SmallCNN 그대로
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
        self.cnn_over = SmallCNN(feat)
        self.head = nn.Sequential(
            nn.Linear(feat * 2 + proprio_dim, 512), nn.ReLU(),
            nn.Linear(512, 512), nn.ReLU(),
            nn.Linear(512, 256), nn.ReLU(),
            nn.Linear(256, action_dim),
        )

    def forward(self, wrist_img, over_img, proprio):
        fw = self.cnn_wrist(wrist_img)
        fo = self.cnn_over(over_img)
        return self.head(torch.cat([fw, fo, proprio], dim=1))


policy = VisionBCPolicy(PROPRIO_DIM, ACTION_DIM).to(device)
optimizer = torch.optim.Adam(policy.parameters(), lr=LR)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=40, gamma=0.5)
loss_fn = nn.MSELoss()
print(f"[bc_train] device={device} | params={sum(p.numel() for p in policy.parameters()):,}")

writer = None
try:
    from torch.utils.tensorboard import SummaryWriter
    writer = SummaryWriter(os.path.join(OUT_DIR, "tb"))
except Exception as exc:
    print(f"[bc_train] tensorboard 없음 ({exc.__class__.__name__}) — CSV만 남긴다")

csv_path = os.path.join(OUT_DIR, "log.csv")
csv_f = open(csv_path, "w", newline="", encoding="utf-8")
csv_w = csv.writer(csv_f)
csv_w.writerow(["epoch", "train_loss", "val_loss", "lr", "sec"])


def save_ckpt(path, epoch, val_loss):
    torch.save({
        "epoch": int(epoch), "policy": policy.state_dict(),
        "proprio_dim": PROPRIO_DIM, "action_dim": ACTION_DIM,
        "chunk_h": CHUNK_H, "joint_dim": JOINT_DIM, "img_size": IMG_SIZE,
        "fps": FPS, "repo_id": REPO_ID, "val_loss": float(val_loss),
        "norm_stats": norm_stats,
    }, path)


# ============================================================================
# 4. 학습 루프
# ============================================================================
train_idx_t = torch.from_numpy(train_idx).to(device)
val_idx_t = torch.from_numpy(val_idx).to(device)
print(f"\n[bc_train] 학습 시작 ({EPOCHS} 에폭, "
      f"{int(np.ceil(len(train_idx)/BATCH_SIZE))} iter/에폭)\n", flush=True)

best_val = float("inf")
for epoch in range(1, EPOCHS + 1):
    t_start = time.time()
    policy.train()
    perm = train_idx_t[torch.randperm(len(train_idx_t), device=device)]
    tr = []
    for i in range(0, len(perm), BATCH_SIZE):
        w, o, pro, tgt = make_batch(perm[i:i + BATCH_SIZE])
        loss = loss_fn(policy(w, o, pro), tgt)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
        optimizer.step()
        tr.append(loss.item())
    scheduler.step()

    policy.eval()
    va = []
    with torch.no_grad():
        for i in range(0, len(val_idx_t), BATCH_SIZE):
            w, o, pro, tgt = make_batch(val_idx_t[i:i + BATCH_SIZE])
            va.append(loss_fn(policy(w, o, pro), tgt).item())

    tl, vl = float(np.mean(tr)), float(np.mean(va))
    dt = time.time() - t_start
    lr_now = optimizer.param_groups[0]["lr"]
    csv_w.writerow([epoch, tl, vl, lr_now, round(dt, 2)])
    csv_f.flush()
    if writer:
        writer.add_scalar("Loss/train", tl, epoch)
        writer.add_scalar("Loss/val", vl, epoch)
    if epoch % 5 == 0 or epoch == 1:
        print(f"Epoch {epoch:3d}/{EPOCHS} | train {tl:.6f} | val {vl:.6f} | "
              f"lr {lr_now:.2e} | {dt:.1f}s", flush=True)

    if vl < best_val:
        best_val = vl
        save_ckpt(os.path.join(OUT_DIR, "checkpoints", "best.pt"), epoch, vl)
    if epoch % SAVE_EVERY == 0 or epoch == EPOCHS:
        save_ckpt(os.path.join(OUT_DIR, "checkpoints", f"epoch_{epoch:03d}.pt"), epoch, vl)
    save_ckpt(os.path.join(OUT_DIR, "checkpoints", "last.pt"), epoch, vl)

csv_f.close()
if writer:
    writer.close()
print(f"\n[bc_train] 완료. best val {best_val:.6f} | 체크포인트 {OUT_DIR}\\checkpoints")
print("[bc_train] val loss는 순위 참고용일 뿐이다 — 성공률은 평가로만 확인된다:")
print("           python bc_policy_server.py  +  C:\\isaacsim\\python.bat -u eval_act_v5_client.py")
