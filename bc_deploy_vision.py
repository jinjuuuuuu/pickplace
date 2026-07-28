# bc_deploy_vision.py  (Isaac Sim 5.1, Windows standalone)
# ---------------------------------------------------------------------------
# [업데이트 완료] 프레임 스태킹(기억력) 탑재 버전!
# 카메라 깜빡임(동기화 버그)을 방어하기 위해 과거 3프레임을 겹쳐서 판단합니다.
# ---------------------------------------------------------------------------

import os, sys, random
import numpy as np
import collections
import torch
import torch.nn as nn

# ============================================================================
# ⚙️ Isaac Sim 초기화 (가장 먼저 실행되어야 함)
# ============================================================================
from isaacsim import SimulationApp
HEADLESS      = False    
RENDER        = True     
simulation_app = SimulationApp({"renderer": "RayTracedLighting", "headless": HEADLESS})

import omni.usd
from pxr import UsdGeom, Gf, UsdLux
from isaacsim.core.api import World
from isaacsim.core.api.objects import DynamicCuboid
from isaacsim.core.utils.types import ArticulationAction
from isaacsim.robot.manipulators.examples.franka import Franka
from isaacsim.sensors.camera import Camera

try:
    import cv2
    def _resize(img, s): return cv2.resize(img, (s, s), interpolation=cv2.INTER_AREA)
except Exception:
    from PIL import Image
    def _resize(img, s): return np.asarray(Image.fromarray(img).resize((s, s)))

# ============================================================================
# === 설정 (경로 주의!) ========================================================
# ============================================================================
# 🔥 주말에 학습하신 스태킹 모델 이름으로 변경했습니다!
POLICY_PATH     = r"C:\Users\user\Desktop\claude_jetbot\bc_policy_vision_resnet_stacked.pt"
NORM_STATS_PATH = r"C:\Users\user\Desktop\claude_jetbot\bc_policy_vision_resnet_stacked_norm_stats.pt"

MAX_STEPS     = 2500
REPLAN_EVERY  = 60
CUBE_POS   = [0.40, 0.15, 0.025]
TARGET_POS = [0.50, -0.15, 0.025]
RANDOMIZE_CUBE_POS = False          
CUBE_X_RANGE = (0.30, 0.55); CUBE_Y_RANGE = (-0.25, 0.25)
RANDOMIZE_CUBE_COLOR = True         
RANDOMIZE_LIGHT      = True
SUCCESS_XY_TOL  = 0.08
SUCCESS_MIN_LIFT = 0.04
WRIST_CAM_PRIM = "/World/Franka/panda_hand/WristCam/Camera"
OVER_CAM_PRIM  = "/World/OverheadCam/Camera"
OVERHEAD_CAM_POS = [0.4, 0.15, 1.5]
CAP_W, CAP_H   = 160, 120            

# ============================================================================
# 1. 🧠 프레임 스태킹 전용 신경망 (설계도)
# ============================================================================
from torchvision.models import resnet18

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
    def __init__(self, in_channels, out_dim=128):
        super().__init__()
        resnet = resnet18(weights=None)
        # 🔥 여기서 in_channels를 받아 여러 장의 사진을 한 번에 볼 수 있게 변경!
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
    def __init__(self, proprio_dim, action_dim, obs_seq=3, feat=128):
        super().__init__()
        self.cnn_wrist = ResNet18Backbone(in_channels=3 * obs_seq, out_dim=feat)
        self.cnn_over  = ResNet18Backbone(in_channels=3 * obs_seq, out_dim=feat)
        self.head = nn.Sequential(
            nn.Linear(feat*2 + proprio_dim * obs_seq, 512), nn.ReLU(),
            nn.Linear(512, 512), nn.ReLU(),
            nn.Linear(512, 256), nn.ReLU(),
            nn.Linear(256, action_dim),
        )
    def forward(self, wrist_img, over_img, proprio):
        fw = self.cnn_wrist(wrist_img)
        fo = self.cnn_over(over_img)
        return self.head(torch.cat([fw, fo, proprio], dim=1))

# ============================================================================
# 2. 정책 + 정규화 로드
# ============================================================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[vis_deploy] {device} 환경에서 정책 로드 중...")

ckpt = torch.load(POLICY_PATH, map_location=device, weights_only=False)

# 🔥 .pt 파일에 메타데이터가 없을 경우를 대비해 우리가 아는 값으로 강제 고정합니다!
PROPRIO_DIM = 18
CHUNK_H = 60
JOINT_DIM = 9
ACTION_DIM = CHUNK_H * JOINT_DIM  # 540
IMG_SIZE = 84                     # ResNet 입력 사이즈 (만약 에러 시 120으로 변경)
OBS_SEQ = 3                       # 기억력(스태킹) 프레임 수

policy = VisionBCPolicy(PROPRIO_DIM, ACTION_DIM, obs_seq=OBS_SEQ).to(device)
policy.load_state_dict(ckpt["policy"]); policy.eval()

norm = torch.load(NORM_STATS_PATH, map_location="cpu", weights_only=False)
pro_mean = norm["proprio_mean"].numpy(); pro_std = norm["proprio_std"].numpy()
act_mean = norm["act_mean"].numpy();     act_std = norm["act_std"].numpy()
START_POSE = norm["start_pose"].numpy() if "start_pose" in norm else \
             np.array([0.0,-0.3,0.0,-2.5,0.0,2.2,0.8,0.04,0.04], dtype=np.float32)

print(f"[vis_deploy] 로드 완료 (epoch {ckpt['epoch']}, H={CHUNK_H}, img={IMG_SIZE}, obs_seq={OBS_SEQ})")

# 🔥 과거 프레임을 기억할 큐(Queue) 생성
wrist_queue = collections.deque(maxlen=OBS_SEQ)
over_queue  = collections.deque(maxlen=OBS_SEQ)
pro_queue   = collections.deque(maxlen=OBS_SEQ)

# ============================================================================
# 3. Scene (수집과 동일 구성)
# ============================================================================
world  = World(stage_units_in_meters=1.0)
world.scene.add_default_ground_plane()
stage  = omni.usd.get_context().get_stage()
franka = world.scene.add(Franka(prim_path="/World/Franka", name="franka"))

cube_pos0 = np.array(CUBE_POS, dtype=float)
if RANDOMIZE_CUBE_POS:
    cube_pos0 = np.array([random.uniform(*CUBE_X_RANGE), random.uniform(*CUBE_Y_RANGE), 0.025])
cube = world.scene.add(DynamicCuboid(
    prim_path="/World/PickCube", name="pick_cube",
    position=cube_pos0, size=0.05, color=np.array([0.8,0.2,0.1]), mass=0.1))

marker = stage.DefinePrim("/World/PlaceMarker", "Cylinder")
UsdGeom.Cylinder(marker).GetRadiusAttr().Set(0.025)
UsdGeom.Cylinder(marker).GetHeightAttr().Set(0.002)
UsdGeom.XformCommonAPI(marker).SetTranslate(Gf.Vec3d(*TARGET_POS))
UsdGeom.Gprim(marker).GetDisplayColorAttr().Set([Gf.Vec3f(0.1, 0.8, 0.1)])

ee_path = "/World/Franka/panda_hand"
wx = stage.DefinePrim(ee_path + "/WristCam", "Xform")
wc = UsdGeom.Camera.Define(stage, WRIST_CAM_PRIM)
wc.GetFocalLengthAttr().Set(16.0)
wc.GetHorizontalApertureAttr().Set(20.955)
wc.GetVerticalApertureAttr().Set(15.716)
wc.GetClippingRangeAttr().Set(Gf.Vec2f(0.05, 100.0))
UsdGeom.XformCommonAPI(wx).SetTranslate(Gf.Vec3d(0.15, 0.0, 0.0)) 
UsdGeom.XformCommonAPI(wx).SetRotate(Gf.Vec3f(-45, 179.9, -89.9))

ox = stage.DefinePrim("/World/OverheadCam", "Xform")
oc = UsdGeom.Camera.Define(stage, OVER_CAM_PRIM)
oc.GetFocalLengthAttr().Set(24.0)
oc.GetHorizontalApertureAttr().Set(20.955)
oc.GetVerticalApertureAttr().Set(15.716)
oc.GetClippingRangeAttr().Set(Gf.Vec2f(0.05, 100.0))
UsdGeom.XformCommonAPI(ox).SetTranslate(Gf.Vec3d(*OVERHEAD_CAM_POS))
UsdGeom.XformCommonAPI(ox).SetRotate(Gf.Vec3f(0.0, 0.0, -89.9))

_dome = UsdLux.DomeLight.Define(stage, "/World/DR_DomeLight")
def randomize_domain():
    if RANDOMIZE_CUBE_COLOR:
        try: cube.get_applied_visual_material().set_color(
            np.array([random.uniform(0,1),random.uniform(0,1),random.uniform(0,1)]))
        except Exception: pass
    if RANDOMIZE_LIGHT:
        try: _dome.GetIntensityAttr().Set(random.uniform(500.0,2500.0))
        except Exception: pass

wrist_cam = Camera(prim_path=WRIST_CAM_PRIM, resolution=(CAP_W, CAP_H))
over_cam  = Camera(prim_path=OVER_CAM_PRIM,  resolution=(CAP_W, CAP_H))

world.reset()
randomize_domain()
wrist_cam.initialize(); over_cam.initialize()
for _ in range(15):
    world.step(render=RENDER)

num_dof = len(franka.get_joint_positions())
finger_idx = np.array([num_dof-2, num_dof-1], dtype=int)
for _ in range(200):
    franka.apply_action(ArticulationAction(joint_positions=np.full(2,0.04), joint_indices=finger_idx))
    world.step(render=RENDER)
start_full = np.array(START_POSE, dtype=float)
if len(start_full) < num_dof:
    cur = np.array(franka.get_joint_positions(), dtype=float); cur[:len(start_full)] = start_full; start_full = cur
for _ in range(60):
    franka.apply_action(ArticulationAction(joint_positions=start_full))
    world.step(render=RENDER)
print(f"[vis_deploy] 시작 자세 이동 완료. 팔={np.round(start_full[:7],3)}")

target_pos = np.array(TARGET_POS, dtype=np.float32)

# ============================================================================
# 4. 관측(이미지+관절) + 예측
# ============================================================================
def grab(cam):
    try:
        rgba = cam.get_rgba()
        if rgba is None or getattr(rgba,"ndim",0)!=3 or rgba.shape[0] < 2:
            return np.zeros((IMG_SIZE,IMG_SIZE,3), np.uint8)
        img = _resize(rgba[:,:,:3].astype(np.uint8), IMG_SIZE)
        return np.ascontiguousarray(img, dtype=np.uint8)
    except Exception:
        return np.zeros((IMG_SIZE,IMG_SIZE,3), np.uint8)

def build_inputs():
    jp = np.array(franka.get_joint_positions(), dtype=np.float32)
    jv = np.clip(franka.get_joint_velocities(), -50.0, 50.0).astype(np.float32)
    proprio = np.concatenate([jp, jv])[:PROPRIO_DIM]
    w = grab(wrist_cam); o = grab(over_cam)
    return proprio, w, o, jp

def predict_chunk(stacked_wrist, stacked_over, stacked_pro):
    with torch.no_grad():
        out = policy(stacked_wrist, stacked_over, stacked_pro).squeeze(0).cpu().numpy()
    chunk = out.reshape(CHUNK_H, JOINT_DIM) * act_std + act_mean
    for h in range(CHUNK_H):     
        chunk[h,7:9] = 0.0 if chunk[h,7] < 0.035 else 0.04
    return chunk

# ============================================================================
# 5. Receding-horizon 제어 루프 (프레임 스태킹 적용)
# ============================================================================
print("\n[vis_deploy] 🚀 비전 정책(프레임 스태킹)으로 Franka 제어 시작!\n")
max_lift = 0.025
step = 0
success = False
chunk = None

while step < MAX_STEPS:
    proprio, w, o, jp = build_inputs()

    # 텐서 변환
    pn = (proprio - pro_mean) / pro_std
    wt = torch.FloatTensor(w.astype(np.float32).transpose(2,0,1)/255.0).unsqueeze(0)
    ot = torch.FloatTensor(o.astype(np.float32).transpose(2,0,1)/255.0).unsqueeze(0)
    pt = torch.FloatTensor(pn).unsqueeze(0)

    # 🔥 큐(Queue)는 매 시뮬레이션 스텝마다 갱신해야 한다. 학습 때는 t-2,t-1,t
    # (한 스텝 간격) 프레임을 스태킹했는데, 이전에는 이 큐 갱신을 예측(chunk) 주기
    # (REPLAN_EVERY=60스텝)에 한 번씩만 했다. 그러면 모델이 보는 "과거 프레임"이
    # 실제로는 60스텝(약 2초)씩 떨어진, 학습 때 전혀 본 적 없는 조합이 되어버려서
    # 예측이 튀고 팔이 이상하게 움직인다.
    if len(wrist_queue) == 0:
        for _ in range(OBS_SEQ):
            wrist_queue.append(wt)
            over_queue.append(ot)
            pro_queue.append(pt)
    else:
        wrist_queue.append(wt)
        over_queue.append(ot)
        pro_queue.append(pt)

    # 정책 추론(청크 예측)은 기존과 동일하게 REPLAN_EVERY 스텝마다 한 번만 수행한다.
    if chunk is None or step % REPLAN_EVERY == 0:
        stacked_wrist = torch.cat(list(wrist_queue), dim=1).to(device)
        stacked_over  = torch.cat(list(over_queue), dim=1).to(device)
        stacked_pro   = torch.cat(list(pro_queue), dim=1).to(device)
        chunk = predict_chunk(stacked_wrist, stacked_over, stacked_pro)

    gt_cube, _ = cube.get_world_pose(); gt_cube = np.array(gt_cube, dtype=np.float32)
    max_lift = max(max_lift, float(gt_cube[2]))
    cube_target_xy = float(np.linalg.norm(gt_cube[:2] - target_pos[:2]))
    gripper_open = jp[7] > 0.035

    if step % 120 == 0:
        print(f"[vis_deploy] step {step:4d} | 큐브(정답) {np.round(gt_cube,3)} | 목표까지 {cube_target_xy:.3f}m | 최대들림 {max_lift:.3f}")

    if cube_target_xy < SUCCESS_XY_TOL and max_lift > (0.025 + SUCCESS_MIN_LIFT) and gripper_open:
        success = True
        print(f"\n[vis_deploy] 🎉 성공! 목표 오차 {cube_target_xy:.3f}m (최대 들림 {max_lift:.3f}m)")
        for _ in range(120): world.step(render=RENDER)
        break

    h = step % REPLAN_EVERY
    franka.apply_action(ArticulationAction(joint_positions=chunk[h].astype(float)))
    world.step(render=RENDER)
    step += 1

gt_cube, _ = cube.get_world_pose(); gt_cube = np.array(gt_cube, dtype=np.float32)
final_err = float(np.linalg.norm(gt_cube[:2] - target_pos[:2]))
print(f"\n[vis_deploy] 종료 | 성공={success} | 최종 목표오차={final_err:.3f}m | 최대들림={max_lift:.3f}m")
simulation_app.close()