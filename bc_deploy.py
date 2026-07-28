# bc_deploy.py  (Isaac Sim 5.1, Windows standalone) — Action Chunking 버전
# ---------------------------------------------------------------------------
# 학습된 chunk 정책으로 PickPlaceController 없이 Franka를 직접 제어한다.
#
# [동작 방식]
#   정책이 obs 1개로 미래 H스텝의 '절대 관절 목표 시퀀스'를 한 번에 예측한다.
#   그 중 앞쪽 REPLAN_EVERY 스텝만 실행하고, 새 obs로 다시 예측한다(receding horizon).
#   -> 큐브 위치에 따라 조향/하강/파지하는 긴 동작을 표현 가능.
#
# Run:
#   "C:\isaacsim\python.bat" "C:\Users\user\Desktop\claude_jetbot\bc_deploy.py"
# ---------------------------------------------------------------------------

# === 설정 ===================================================================
POLICY_PATH    = r"C:\Users\user\Desktop\claude_jetbot\bc_data_v3\bc_policy.pt"
NORM_STATS_PATH= r"C:\Users\user\Desktop\claude_jetbot\bc_data_v3\bc_policy_norm_stats.pt"

HEADLESS       = False
ISAACSIM_PATH  = r"C:\isaacsim"
FASTDDS_XML    = r"C:\.ros\fastdds.xml"
MAX_STEPS      = 2500

# receding horizon: 예측한 H스텝 중 앞 REPLAN_EVERY 스텝만 실행 후 재예측.
# 작을수록 폐루프(반응성↑, 느림). 클수록 개루프(빠름, 드리프트↑). H의 1/4~1/2 권장.
REPLAN_EVERY   = 60

CUBE_POS    = [0.40, 0.15, 0.025]
TARGET_POS  = [0.50, -0.15, 0.025]
# =============================================================================

import os, sys
import numpy as np
import torch
import torch.nn as nn

if sys.platform == "win32":
    os.environ.setdefault("ROS_DISTRO",        "humble")
    os.environ.setdefault("RMW_IMPLEMENTATION","rmw_fastrtps_cpp")
    os.environ.setdefault("ROS_DOMAIN_ID",     "0")
    if os.path.exists(FASTDDS_XML):
        os.environ.setdefault("FASTRTPS_DEFAULT_PROFILES_FILE", FASTDDS_XML)
    bridge_lib = os.path.join(ISAACSIM_PATH, "exts",
                              "isaacsim.ros2.bridge", "humble", "lib")
    if os.path.isdir(bridge_lib):
        os.environ["PATH"] = bridge_lib + os.pathsep + os.environ.get("PATH", "")
        try:
            os.add_dll_directory(bridge_lib)
        except Exception:
            pass

from isaacsim import SimulationApp
simulation_app = SimulationApp({"renderer": "RayTracedLighting", "headless": HEADLESS})

import omni.usd
from pxr import UsdGeom, Gf

from isaacsim.core.api import World
from isaacsim.core.api.objects import DynamicCuboid
from isaacsim.core.utils import extensions
from isaacsim.core.utils.types import ArticulationAction
from isaacsim.robot.manipulators.examples.franka import Franka

extensions.enable_extension("isaacsim.ros2.bridge")
simulation_app.update()

# ============================================================================
# 1. 신경망 (bc_train.py 와 완전히 동일)
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

# ============================================================================
# 2. 정책 + 정규화 로드
# ============================================================================
print("[deploy] 정책 로드 중...")
ckpt = torch.load(POLICY_PATH, map_location="cpu", weights_only=False)
OBS_DIM    = ckpt["obs_dim"]
ACTION_DIM = ckpt["action_dim"]
CHUNK_H    = ckpt["chunk_h"]
JOINT_DIM  = ckpt["joint_dim"]

policy = BCChunkPolicy(OBS_DIM, ACTION_DIM)
policy.load_state_dict(ckpt["policy"])
policy.eval()
print(f"[deploy] 정책 로드 완료 (epoch {ckpt['epoch']}, val loss {ckpt['val_loss']:.6f}, "
      f"H={CHUNK_H}, replan={REPLAN_EVERY})")

norm = torch.load(NORM_STATS_PATH, map_location="cpu", weights_only=False)
obs_mean = norm["obs_mean"].numpy()
obs_std  = norm["obs_std"].numpy()
act_mean = norm["act_mean"].numpy()     # (9,)
act_std  = norm["act_std"].numpy()      # (9,)
START_POSE = norm["start_pose"].numpy() if "start_pose" in norm else None
print("[deploy] 정규화 통계 로드 완료")

# ============================================================================
# 3. Scene 구성
# ============================================================================
world  = World(stage_units_in_meters=1.0)
world.scene.add_default_ground_plane()
stage  = omni.usd.get_context().get_stage()

franka = world.scene.add(Franka(prim_path="/World/Franka", name="franka"))

cube = world.scene.add(DynamicCuboid(
    prim_path="/World/PickCube", name="pick_cube",
    position=np.array(CUBE_POS), size=0.05,
    color=np.array([0.8, 0.2, 0.1]), mass=0.1,
))

marker = stage.DefinePrim("/World/PlaceMarker", "Cylinder")
UsdGeom.Cylinder(marker).GetRadiusAttr().Set(0.025)
UsdGeom.Cylinder(marker).GetHeightAttr().Set(0.002)
UsdGeom.XformCommonAPI(marker).SetTranslate(Gf.Vec3d(*TARGET_POS))
UsdGeom.Gprim(marker).GetDisplayColorAttr().Set([Gf.Vec3f(0.1, 0.8, 0.1)])

world.reset()

# ---- 워밍업: 그리퍼 열기 ----------------------------------------------------
FINGER_OPEN = 0.04
num_dof = len(franka.get_joint_positions())
finger_idx = np.array([num_dof - 2, num_dof - 1], dtype=int)
print("[deploy] 워밍업(그리퍼 열기)...")
for _ in range(200):
    franka.apply_action(ArticulationAction(
        joint_positions=np.full(2, FINGER_OPEN, dtype=float),
        joint_indices=finger_idx))
    world.step(render=not HEADLESS)
print(f"[deploy] 워밍업 완료. 손가락: "
      f"{np.round(np.array(franka.get_joint_positions())[finger_idx],4)}")

# ---- 수집과 동일한 고정 시작 자세로 이동 -----------------------------------
if START_POSE is not None:
    start_full = np.array(START_POSE, dtype=float)
else:
    start_full = np.array([0.0,-0.3,0.0,-2.5,0.0,2.2,0.8,0.04,0.04], dtype=float)
if len(start_full) < num_dof:
    cur = np.array(franka.get_joint_positions(), dtype=float)
    cur[:len(start_full)] = start_full
    start_full = cur
print(f"[deploy] 시작 자세로 이동: 팔={np.round(start_full[:7],3)}")
for _ in range(60):
    franka.apply_action(ArticulationAction(joint_positions=start_full))
    world.step(render=not HEADLESS)
print(f"[deploy] 이동 완료. 현재 팔: {np.round(np.array(franka.get_joint_positions())[:7],3)}")

target_pos = np.array(TARGET_POS, dtype=np.float32)

# ============================================================================
# 4. 관측 구성 (수집 get_observation 과 동일한 순서/구성!)
# ============================================================================
def build_obs():
    joint_pos = np.array(franka.get_joint_positions(), dtype=np.float32)
    joint_vel = np.clip(franka.get_joint_velocities(), -50.0, 50.0).astype(np.float32)
    cube_world_pos, _ = cube.get_world_pose()
    cube_world_pos = np.array(cube_world_pos, dtype=np.float32)
    try:
        ee_pos, _ = franka.end_effector.get_world_pose()
        ee_pos = np.array(ee_pos, dtype=np.float32)
    except Exception:
        hand = stage.GetPrimAtPath("/World/Franka/panda_hand")
        xf = UsdGeom.Xformable(hand).ComputeLocalToWorldTransform(0)
        t = xf.ExtractTranslation()
        ee_pos = np.array([t[0], t[1], t[2]], dtype=np.float32)
    cube_rel   = cube_world_pos - ee_pos
    target_rel = target_pos     - ee_pos
    obs = np.concatenate([joint_pos, joint_vel, cube_rel, target_rel, cube_world_pos])
    return obs, joint_pos, cube_world_pos, cube_rel

def predict_chunk(obs):
    obs_norm = (obs - obs_mean) / obs_std
    with torch.no_grad():
        out = policy(torch.FloatTensor(obs_norm).unsqueeze(0)).squeeze(0).numpy()
    chunk = out.reshape(CHUNK_H, JOINT_DIM)
    chunk = chunk * act_std + act_mean        # 절대 관절 목표로 복원 (브로드캐스트)
    
    # === [추가된 부분] 그리퍼 값을 확실하게 열거나 닫도록 강제 보정 ===
    # 손가락 관절 인덱스 (7, 8번)
    # 예측값이 0.035 보다 작으면 무조건 0.0(꽉 닫힘)으로, 크면 0.04(활짝 열림)로 만듭니다.
    GRIPPER_THRESHOLD = 0.035
    for h in range(CHUNK_H):
        if chunk[h, 7] < GRIPPER_THRESHOLD:
            chunk[h, 7:9] = 0.0   # 꽉 닫기 (물리 엔진이 큐브 크기에 맞춰 멈춰줌)
        else:
            chunk[h, 7:9] = 0.04  # 활짝 열기
    # =================================================================
            
    return chunk

# ============================================================================
# 5. Receding-horizon chunk 제어 루프
# ============================================================================
print(f"\n[deploy] 신경망(chunk)으로 Franka 제어 시작!\n")
step = 0
while step < MAX_STEPS:
    obs, joint_pos, cube_world_pos, cube_rel = build_obs()
    chunk = predict_chunk(obs)               # (H, 9) 절대 목표

    # === [추가/수정] 성공 판정 및 3초 대기 후 복귀 로직 ===
    cube_target_dist = np.linalg.norm(cube_world_pos[:2] - target_pos[:2])
    # 큐브가 타겟 5cm 이내에 있고, 그리퍼가 충분히 열렸다면(0.035 이상) 성공으로 간주
    if cube_target_dist < 0.05 and joint_pos[7] > 0.035:
        print(f"\n[deploy] 🎉 큐브 배치 완료! (목표 오차: {cube_target_dist:.3f}m)")
        print("[deploy] 3초 대기 중...")
        
        # 1. 3초 대기 (Isaac Sim 기본 주기가 대략 60Hz이므로 180스텝)
        for _ in range(180):
            world.step(render=not HEADLESS)
            
        print("[deploy] 초기 시작 자세로 복귀합니다...")
        
        # 2. 처음 자세(start_full)로 부드럽게 복귀 (대략 2초/120스텝 소요)
        for _ in range(120):
            franka.apply_action(ArticulationAction(joint_positions=start_full))
            world.step(render=not HEADLESS)
            
        print("[deploy] 복귀 완료. 프로그램을 깔끔하게 종료합니다.")
        break  # while 루프 강제 종료
    # =======================================================

    if step % 120 == 0 or step == 0:
        print(f"[deploy] step {step:4d} | 큐브높이 {float(cube_world_pos[2]):.3f}m "
              f"| 손→큐브 {np.linalg.norm(cube_rel):.3f}m")
        print(f"          현재 팔 : {np.round(joint_pos[:7],3)}")
        print(f"          목표 팔(끝) : {np.round(chunk[-1,:7],3)}")
        print(f"          손가락 현재/목표(끝): "
              f"{np.round(joint_pos[7:9],3)} / {np.round(chunk[-1,7:9],3)}")

    # 앞 REPLAN_EVERY 스텝만 실행
    for h in range(min(REPLAN_EVERY, CHUNK_H)):
        if step >= MAX_STEPS:
            break
        franka.apply_action(ArticulationAction(joint_positions=chunk[h].astype(float)))
        world.step(render=not HEADLESS)
        step += 1

print("\n[deploy] 완료!")
simulation_app.close()