# act_deploy_isaac.py  (Isaac Sim 5.1, Windows standalone)
# ---------------------------------------------------------------------------
# 학습된 ACT 정책을 Isaac Sim에 얹어 카메라 영상으로 Franka를 직접 제어한다.
#   - 매 스텝 손목+천장 카메라 RGB와 관절각을 정책에 넣어 미래 행동 chunk를 예측
#   - temporal aggregation으로 매 스텝 예측을 앙상블해 부드럽게 실행
#
# 사전: act_train.py 로 CHECKPOINT_DIR/policy_best.ckpt 와 dataset_stats.pkl 생성.
#
# 실행:
#   "C:\isaacsim\python.bat" "C:\Users\user\Desktop\claude_jetbot\ACT\act_deploy_isaac.py"
# ---------------------------------------------------------------------------
import os, sys, pickle
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config as C

HEADLESS = False

from isaacsim import SimulationApp
simulation_app = SimulationApp({"renderer": "RayTracedLighting", "headless": HEADLESS})

import torch
from einops import rearrange
import omni.usd
from pxr import UsdGeom, Gf

from isaacsim.core.api import World
from isaacsim.core.api.objects import DynamicCuboid
from isaacsim.core.utils.types import ArticulationAction
from isaacsim.robot.manipulators.examples.franka import Franka
from act_cameras import create_cameras, init_cameras, grab_rgb

# detr 모델 빌드시 argparse가 sys.argv를 읽으므로 비워준다.
sys.argv = [sys.argv[0]]
from act_dataset import make_policy

device = C.device

# ============================================================================
# 1. 정책 + 정규화 통계 로드
# ============================================================================
ckpt_path  = os.path.join(C.CHECKPOINT_DIR, "policy_best.ckpt")
stats_path = os.path.join(C.CHECKPOINT_DIR, "dataset_stats.pkl")
print(f"[deploy] 정책 로드: {ckpt_path}")
policy = make_policy("ACT", C.build_policy_config())
policy.load_state_dict(torch.load(ckpt_path, map_location=device))
policy.to(device).eval()

with open(stats_path, "rb") as f:
    stats = pickle.load(f)
pre_process  = lambda q: (q - stats["qpos_mean"]) / stats["qpos_std"]
post_process = lambda a: a * stats["action_std"] + stats["action_mean"]
print("[deploy] dataset_stats 로드 완료")

NUM_Q = C.CHUNK_SIZE

# ============================================================================
# 2. Scene (수집과 동일 구성 + 카메라)
# ============================================================================
world = World(stage_units_in_meters=1.0)
world.scene.add_default_ground_plane()
stage = omni.usd.get_context().get_stage()

franka = world.scene.add(Franka(prim_path="/World/Franka", name="franka"))
cube = world.scene.add(DynamicCuboid(
    prim_path="/World/PickCube", name="pick_cube",
    position=np.array(C.EVAL_CUBE_POS), size=0.05,
    color=np.array([0.8, 0.2, 0.1]), mass=0.1))
marker = stage.DefinePrim("/World/PlaceMarker", "Cylinder")
UsdGeom.Cylinder(marker).GetRadiusAttr().Set(0.025)
UsdGeom.Cylinder(marker).GetHeightAttr().Set(0.002)
UsdGeom.XformCommonAPI(marker).SetTranslate(Gf.Vec3d(*C.EVAL_TARGET_POS))
UsdGeom.Gprim(marker).GetDisplayColorAttr().Set([Gf.Vec3f(0.1, 0.8, 0.1)])

# 고정 카메라 (Camera 클래스가 프림 직접 생성)
CAMS = create_cameras(C)

world.reset()
init_cameras(CAMS, world, simulation_app, warmup=30)

num_dof = franka.num_dof
finger_idx = np.array([num_dof - 2, num_dof - 1], dtype=int)

# 시작 자세로 이동 (수집과 동일)
start_full = np.array(franka.get_joint_positions(), dtype=float)
start_full[:7] = np.array(C.START_POSE, dtype=float)
start_full[finger_idx] = C.FINGER_OPEN
for _ in range(60):
    franka.apply_action(ArticulationAction(joint_positions=start_full))
    world.step(render=True)
print(f"[deploy] 시작 자세 이동 완료: {np.round(np.array(franka.get_joint_positions())[:7],3)}")


def get_image_tensor():
    imgs = []
    for cn in C.CAMERA_NAMES:
        im = rearrange(grab_rgb(CAMS[cn], C), "h w c -> c h w")
        imgs.append(im)
    arr = np.stack(imgs, axis=0)                       # (num_cam,3,H,W)
    return torch.from_numpy(arr / 255.0).float().to(device).unsqueeze(0)


def read_qpos():
    jp = np.array(franka.get_joint_positions(), dtype=np.float32)
    grip = 1.0 if float(jp[finger_idx[0]]) < C.GRASP_FINGER_THRESH else 0.0
    return np.concatenate([jp[:7], [grip]]).astype(np.float32)


# ============================================================================
# 3. 추론 루프 (temporal aggregation)
# ============================================================================
T = C.DEPLOY_MAX_STEPS
if C.TEMPORAL_AGG:
    all_time_actions = torch.zeros([T, T + NUM_Q, C.STATE_DIM], device=device)
query_freq = 1 if C.TEMPORAL_AGG else NUM_Q

print(f"\n[deploy] ACT 제어 시작 (temporal_agg={C.TEMPORAL_AGG}, chunk={NUM_Q})\n")
all_actions = None
with torch.inference_mode():
    for t in range(T):
        qpos_np = read_qpos()
        qpos = torch.from_numpy(pre_process(qpos_np)).float().to(device).unsqueeze(0)
        image = get_image_tensor()

        if t % query_freq == 0:
            all_actions = policy(qpos, image)          # (1, NUM_Q, 8)

        if C.TEMPORAL_AGG:
            all_time_actions[[t], t:t + NUM_Q] = all_actions
            step_actions = all_time_actions[:, t]
            populated = torch.all(step_actions != 0, axis=1)
            step_actions = step_actions[populated]
            k = 0.01
            w = np.exp(-k * np.arange(len(step_actions)))
            w = w / w.sum()
            w = torch.from_numpy(w.astype(np.float32)).to(device).unsqueeze(1)
            raw = (step_actions * w).sum(dim=0, keepdim=True)
        else:
            raw = all_actions[:, t % query_freq]

        action = post_process(raw.squeeze(0).cpu().numpy())   # (8,)

        # 명령 구성: 팔 7관절 + 그리퍼(이진->손가락 폭)
        cmd = np.array(franka.get_joint_positions(), dtype=float)
        cmd[:7] = action[:7]
        finger = C.FINGER_CLOSE if action[7] > 0.5 else C.FINGER_OPEN
        cmd[finger_idx] = finger
        franka.apply_action(ArticulationAction(joint_positions=cmd))
        world.step(render=not HEADLESS)

        if t % 60 == 0:
            cz = float(cube.get_world_pose()[0][2])
            print(f"[deploy] t={t:3d} | 큐브높이 {cz:.3f}m | "
                  f"그리퍼 {'CLOSE' if action[7]>0.5 else 'OPEN'} | "
                  f"팔목표 {np.round(action[:7],2)}")

print("\n[deploy] 완료!")
simulation_app.close()
