# diag_vision_policy.py — 비전 BC가 '이미지를 실제로 쓰는지' 진단
# 실행 (torch 있는 아무 환경, WSL2 권장):
#   python3 diag_vision_policy.py
# ---------------------------------------------------------------------------
# 두 가지를 본다:
#  (1) 이미지 제거(0)했을 때 예측이 얼마나 바뀌나  → 거의 안 바뀌면 '이미지 무시'
#  (2) 큐브 위치가 다른 여러 에피소드에서 첫 행동이 다양한가 → 다 비슷하면 '평균 재생'
# ---------------------------------------------------------------------------
import os, glob, numpy as np, torch, torch.nn as nn
try: import cv2; _rs=lambda im,s: cv2.resize(im,(s,s),interpolation=cv2.INTER_AREA)
except Exception:
    from PIL import Image; _rs=lambda im,s: np.asarray(Image.fromarray(im).resize((s,s)))

_WIN=r"C:\Users\user\Desktop\claude_jetbot\bc_data_v3"; _WSL="/mnt/c/Users/user/Desktop/claude_jetbot/bc_data_v3"
DIR=_WIN if os.path.isdir(_WIN) else _WSL
CK=os.path.join(DIR,"bc_policy_vision.pt"); NS=os.path.join(DIR,"bc_policy_vision_norm_stats.pt")

from torchvision.models import resnet18
try:
    from torchvision.models import ResNet18_Weights
    _IMAGENET_W = ResNet18_Weights.IMAGENET1K_V1
except Exception:
    _IMAGENET_W = True
USE_PRETRAINED = False
_IMEAN = torch.tensor([0.485,0.456,0.406]).view(1,3,1,1)
_ISTD  = torch.tensor([0.229,0.224,0.225]).view(1,3,1,1)
class SpatialSoftmax(nn.Module):
    def forward(s,x):
        B,C,H,W=x.shape
        a=torch.softmax(x.reshape(B,C,H*W),dim=-1)
        ys,xs=torch.meshgrid(torch.linspace(-1,1,H,device=x.device),torch.linspace(-1,1,W,device=x.device),indexing="ij")
        return torch.cat([(a*xs.reshape(-1)).sum(-1),(a*ys.reshape(-1)).sum(-1)],1)
class ResNetEncoder(nn.Module):
    def __init__(s,o=128):
        super().__init__()
        try: m=resnet18(weights=(_IMAGENET_W if USE_PRETRAINED else None))
        except Exception as e: print("[warn] resnet load fail:",e); m=resnet18(weights=None)
        s.backbone=nn.Sequential(m.conv1,m.bn1,m.relu,m.maxpool,m.layer1,m.layer2,m.layer3)
        s.reduce=nn.Conv2d(256,32,1); s.sm=SpatialSoftmax(); s.fc=nn.Linear(64,o)
    def forward(s,x):
        x=(x-_IMEAN.to(x.device))/_ISTD.to(x.device)
        return torch.relu(s.fc(s.sm(s.reduce(s.backbone(x)))))
class VisionBCPolicy(nn.Module):
    def __init__(s,p,a,f=128):
        super().__init__(); s.cnn_wrist=ResNetEncoder(f); s.cnn_over=ResNetEncoder(f)
        s.head=nn.Sequential(nn.Linear(f*2+p,512),nn.ReLU(),nn.Linear(512,512),nn.ReLU(),
                             nn.Linear(512,256),nn.ReLU(),nn.Linear(256,a))
    def forward(s,w,o,p): return s.head(torch.cat([s.cnn_wrist(w),s.cnn_over(o),p],1))

ck=torch.load(CK,map_location="cpu",weights_only=False); ns=torch.load(NS,map_location="cpu",weights_only=False)
P=ck["proprio_dim"]; A=ck["action_dim"]; IMG=ck["img_size"]; H=ck["chunk_h"]; J=ck["joint_dim"]
pm=ns["proprio_mean"].numpy(); ps=ns["proprio_std"].numpy(); am=ns["act_mean"].numpy(); as_=ns["act_std"].numpy()
pol=VisionBCPolicy(P,A); pol.load_state_dict(ck["policy"]); pol.eval()
print(f"[diag] 로드 완료 epoch={ck['epoch']} val={ck['val_loss']:.6f} P={P} H={H}")

def prep(img): return torch.FloatTensor(_rs(img,IMG).astype(np.float32).transpose(2,0,1)/255.).unsqueeze(0)
def predict(w,o,pro,zero=False):
    wt=torch.zeros(1,3,IMG,IMG) if zero else prep(w)
    ot=torch.zeros(1,3,IMG,IMG) if zero else prep(o)
    pt=torch.FloatTensor(((pro-pm)/ps).astype(np.float32)).unsqueeze(0)
    with torch.no_grad(): out=pol(wt,ot,pt).squeeze(0).numpy()
    return (out.reshape(H,J)*as_+am)

files=sorted(glob.glob(os.path.join(DIR,"episode_*.npz")))[:15]
diffs=[]; c0=[]; c30=[]; c59=[]; cubes=[]; imgs=[]
for f in files:
    d=np.load(f); obs=d["obs"].astype(np.float32); iw=d["images_wrist"]; io=d["images_over"]
    pro=obs[0,:P]
    real=predict(iw[0],io[0],pro,zero=False)
    diffs.append(np.abs(real-predict(iw[0],io[0],pro,zero=True)).mean())
    c0.append(real[0,:7]); c30.append(real[30,:7]); c59.append(real[59,:7])
    cubes.append(d["cube_pos"][:2] if "cube_pos" in d.files else obs[0,24:26])
    imgs.append((iw[0],io[0]))

diffs=np.array(diffs); c0=np.array(c0); c30=np.array(c30); c59=np.array(c59); cubes=np.array(cubes)
print(f"\n[결과 1] 이미지 제거 시 예측 변화량(관절 rad): {diffs.mean():.4f}  (0.01↓ 무시 / 0.1↑ 잘씀)")
print(f"\n[결과 2] 15개 에피소드 예측의 관절별 표준편차 (청크 스텝별):")
print(f"  step 0 : {np.round(c0.std(0),4)}")
print(f"  step 30: {np.round(c30.std(0),4)}")
print(f"  step 59: {np.round(c59.std(0),4)}   ← 뒷부분이 커야 '큐브 따라 조준'")

# [결과 3] 관절값 '고정' + 이미지만 15종으로 바꿔치기 → 순수 이미지가 궤적을 바꾸나?
fixed_pro=np.load(files[0])["obs"][0,:P].astype(np.float32)
sw=[predict(w,o,fixed_pro)[59,:7] for (w,o) in imgs]
sw=np.array(sw)
print(f"\n[결과 3] 관절값 고정 + 이미지만 교체했을 때 step59 표준편차: {np.round(sw.std(0),4)}")
print(f"  → 이게 ~0 이면 '이미지가 궤적을 못 바꿈'(진짜 이미지 무시). 크면 '이미지가 궤적을 좌우'.")
print(f"\n  큐브 위치 범위: x[{cubes[:,0].min():.2f},{cubes[:,0].max():.2f}] y[{cubes[:,1].min():.2f},{cubes[:,1].max():.2f}]")
