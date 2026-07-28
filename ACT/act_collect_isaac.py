# act_collect_isaac.py  (Isaac Sim 5.1, Windows standalone)
# ---------------------------------------------------------------------------
# ACT 학습용 데이터 수집기.
#   - PickPlaceController(전문가)로 pick&place 데모를 생성한다.
#   - 손목(wrist) + 천장(top) 카메라 RGB, 관절각, 행동을 매 스텝 기록한다.
#   - 성공한 에피소드만 EPISODE_LEN 프레임으로 균일 리샘플링해
#     ACT 표준 HDF5 포맷으로 저장한다:
#         /observations/images/top   (T,480,640,3) uint8
#         /observations/images/wrist (T,480,640,3) uint8
#         /observations/qpos         (T,8)  = [관절1..7, 그리퍼(0/1)]
#         /action                    (T,8)  = qpos 와 동일(미래 자세 시퀀스를 학습)
#         attrs['sim'] = True
#
# 실행:
#   "C:\isaacsim\python.bat" "C:\Users\user\Desktop\claude_jetbot\ACT\act_collect_isaac.py"
# ---------------------------------------------------------------------------
import os, sys, random
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config as C

HEADLESS = True   # 카메라 캡처는 headless여도 render=True로 동작한다.

os.makedirs(C.DATASET_DIR, exist_ok=True)

from isaacsim import SimulationApp
simulation_app = SimulationApp({"renderer": "RayTracedLighting", "headless": HEADLESS})

import omni.usd
from pxr import UsdGeom, Gf

from isaacsim.core.api import World
from isaacsim.core.api.objects import DynamicCuboid
from isaacsim.core.utils.types import ArticulationAction
from isaacsim.robot.manipulators.examples.franka import Franka
from isaacsim.robot.manipulators.examples.franka.controllers import PickPlaceController

from act_cameras import create_cameras, init_cameras, grab_rgb


# ============================================================================
def sample_positions():
    while True:
        cx = random.uniform(*C.CUBE_X_RANGE);  cy = random.uniform(*C.CUBE_Y_RANGE)
        tx = random.uniform(*C.TARGET_X_RANGE); ty = random.uniform(*C.TARGET_Y_RANGE)
        if np.hypot(cx - tx, cy - ty) >= C.MIN_DISTANCE:
            return (np.array([cx, cy, C.CUBE_Z]), np.array([tx, ty, C.TARGET_Z]))


def gripper_binary(finger_width):
    return 1.0 if finger_width < C.GRASP_FINGER_THRESH else 0.0


def resample(seq, n):
    """길이 Tr 시퀀스를 n개로 균일 리샘플링(인덱스 기반)."""
    seq = np.asarray(seq)
    Tr = len(seq)
    idx = np.linspace(0, Tr - 1, n).round().astype(int)
    return seq[idx]


# ============================================================================
# Scene
# ============================================================================
world = World(stage_units_in_meters=1.0)
world.scene.add_default_ground_plane()
stage = omni.usd.get_context().get_stage()

franka = world.scene.add(Franka(prim_path="/World/Franka", name="franka"))

cube = world.scene.add(DynamicCuboid(
    prim_path="/World/PickCube", name="pick_cube",
    position=np.array([0.45, 0.0, C.CUBE_Z]), size=0.05,
    color=np.array([0.8, 0.2, 0.1]), mass=0.1))

marker = stage.DefinePrim("/World/PlaceMarker", "Cylinder")
UsdGeom.Cylinder(marker).GetRadiusAttr().Set(0.025)
UsdGeom.Cylinder(marker).GetHeightAttr().Set(0.002)
UsdGeom.Gprim(marker).GetDisplayColorAttr().Set([Gf.Vec3f(0.1, 0.8, 0.1)])

# --- 고정 카메라 생성 (Camera 클래스가 프림 직접 생성: intrinsic 자동 설정) ---
CAMS = create_cameras(C)

world.reset()
init_cameras(CAMS, world, simulation_app, warmup=30)   # 워밍업->initialize->워밍업
print(f"[collect] cameras={list(CAMS)}")

num_dof = franka.num_dof
finger_idx = np.array([num_dof - 2, num_dof - 1], dtype=int)
print(f"[collect] DOF={num_dof}, cameras={list(CAMS)}")

saved = 0
ep_idx = 0
while saved < C.NUM_EPISODES:
    cube_pos, target_pos = sample_positions()
    world.reset()
    cube.set_world_pose(position=cube_pos)
    UsdGeom.XformCommonAPI(marker).SetTranslate(Gf.Vec3d(*target_pos))

    # 그리퍼 열기
    try:
        franka.gripper.open()
    except Exception:
        franka.gripper.apply_action(
            ArticulationAction(joint_positions=franka.gripper.joint_opened_positions))
    for _ in range(20):
        world.step(render=True)
    # 물리 안정화
    for _ in range(60):
        world.step(render=True)
    # 고정 시작 자세
    start_full = np.array(franka.get_joint_positions(), dtype=float)
    start_full[:7] = np.array(C.START_POSE, dtype=float)
    start_full[finger_idx] = C.FINGER_OPEN
    for _ in range(60):
        franka.apply_action(ArticulationAction(joint_positions=start_full))
        world.step(render=True)

    controller = PickPlaceController(name="pp", gripper=franka.gripper,
                                     robot_articulation=franka)

    qpos_seq, img_seq = [], {cn: [] for cn in C.CAMERA_NAMES}
    max_cube_z = C.CUBE_Z
    step = 0
    while step < C.COLLECT_MAX_STEPS:
        jp = np.array(franka.get_joint_positions(), dtype=np.float32)
        finger_w = float(jp[finger_idx[0]])
        qpos = np.concatenate([jp[:7], [gripper_binary(finger_w)]]).astype(np.float32)
        qpos_seq.append(qpos)
        for cn in C.CAMERA_NAMES:
            img_seq[cn].append(grab_rgb(CAMS[cn], C))

        action = controller.forward(picking_position=cube_pos,
                                     placing_position=target_pos,
                                     current_joint_positions=jp)
        franka.apply_action(action)
        world.step(render=True)
        step += 1
        cz = float(cube.get_world_pose()[0][2])
        max_cube_z = max(max_cube_z, cz)
        if controller.is_done():
            break

    # 성공 판정: 큐브가 들렸고 최종 위치가 타겟 근처
    final_cube = np.array(cube.get_world_pose()[0], dtype=np.float32)
    lifted = (max_cube_z - C.CUBE_Z) >= 0.04
    placed = np.linalg.norm(final_cube[:2] - target_pos[:2]) <= 0.06
    if not (lifted and placed) or len(qpos_seq) < 10:
        print(f"[collect] ep {ep_idx} 실패(lifted={lifted},placed={placed}) - 버림")
        ep_idx += 1
        continue

    # 리샘플 + 저장 (h5py 미사용: Isaac Windows 파이썬에서 DLL 충돌 회피 -> .npz)
    qpos_rs = resample(qpos_seq, C.EPISODE_LEN).astype(np.float64)         # (T,8)
    action_rs = qpos_rs.copy()                                            # action = 미래 qpos 시퀀스
    save_kw = {"qpos": qpos_rs, "action": action_rs, "sim": np.array([1], np.int8)}
    for cn in C.CAMERA_NAMES:
        save_kw[f"image_{cn}"] = resample(img_seq[cn], C.EPISODE_LEN).astype(np.uint8)  # (T,480,640,3)
    path = os.path.join(C.DATASET_DIR, f"episode_{saved}.npz")
    np.savez_compressed(path, **save_kw)
    saved += 1
    ep_idx += 1
    print(f"[collect] ✅ episode_{saved-1}.npz 저장 "
          f"(원본 {len(qpos_seq)}스텝 -> {C.EPISODE_LEN}) | 성공 {saved}/{C.NUM_EPISODES}")

print(f"\n[collect] 완료! {saved}개 에피소드 -> {C.DATASET_DIR}")
simulation_app.close()
