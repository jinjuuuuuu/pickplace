# act_hub_deploy_isaac.py  (Isaac Sim 5.1, Windows standalone)
# ---------------------------------------------------------------------------
# 허깅페이스 허브(jamongsteak/act_pickplace_model)에 올라간 LeRobot ACT 정책을
# "그대로" 불러와 Isaac Sim의 Franka pick&place 씬에서 실행한다.
#
# 이 정책은 pick_place_collect.py 로 모은 bc_data_v3 + convert_to_lerobot.py 로
# 만든 LeRobotDataset(jamongsteak/pickplace_vision_v2)으로 학습되었다. 허브의
# config.json 기준 입력/출력은:
#   observation.images.wrist : (3,120,160)  - panda_hand에 부착된 손목 카메라
#   observation.images.over  : (3,120,160)  - 천장 고정 카메라
#   observation.state        : (18,)        - 관절위치(9) + 관절속도(9)
#   action                   : (9,)         - 관절위치 목표(팔7 + 그리퍼핑거2), 절대값
# 따라서 씬/카메라 배치를 pick_place_collect.py와 동일하게 맞춘다.
#
# 최초 실행 시 허브에서 정책 가중치(~210MB)와 정규화(전/후처리) 통계를 자동
# 다운로드해 로컬 캐시(~/.cache/huggingface)에 저장한다. 인터넷 연결 필요.
#
# 실행:
#   "C:\isaacsim\python.bat" "C:\Users\user\Desktop\claude_jetbot\act_hub_deploy_isaac.py"
# ---------------------------------------------------------------------------
import random
import numpy as np

REPO_ID   = "jamongsteak/act_pickplace_model"
TASK      = "pick up the cube and place it on the target"
HEADLESS  = False
MAX_STEPS = 1500
IMG_W, IMG_H = 160, 120
# pick_place_collect.py(학습 데이터 수집)와 동일하게 맞춰야 하는 값.
# 물리/렌더가 충분히 수렴(RTX 디노이저 안정화)하기 전에 정책을 시작하면,
# 학습 때 한 번도 본 적 없는 "덜 수렴된" 이미지로 첫 액션 청크를 예측하게 된다.
WARMUP_STEPS = 200

CUBE_POS   = [0.45, 0.10, 0.025]
TARGET_POS = [0.50, -0.15, 0.025]
START_POSE = [0.0, -0.3, 0.0, -2.5, 0.0, 2.2, 0.8, 0.04, 0.04]

SUCCESS_XY_TOL   = 0.05
SUCCESS_MIN_LIFT = 0.04

WRIST_CAM_PRIM = "/World/Franka/panda_hand/WristCam/Camera"
OVER_CAM_PRIM  = "/World/OverheadCam/Camera"

from isaacsim import SimulationApp
simulation_app = SimulationApp({"renderer": "RayTracedLighting", "headless": HEADLESS})

import torch
import omni.usd
from pxr import UsdGeom, Gf, UsdLux
from isaacsim.core.api import World
from isaacsim.core.api.objects import DynamicCuboid
from isaacsim.core.utils.types import ArticulationAction
from isaacsim.robot.manipulators.examples.franka import Franka
from isaacsim.sensors.camera import Camera

from lerobot.policies.act.modeling_act import ACTPolicy
from lerobot.policies.factory import make_pre_post_processors
from lerobot.utils.control_utils import predict_action
from lerobot.utils.utils import get_safe_torch_device

# ============================================================================
# 1. 허브에서 ACT 정책 + 전/후처리 파이프라인 로드
# ============================================================================
print(f"[deploy] 허깅페이스 허브에서 정책 로드: {REPO_ID}")
policy = ACTPolicy.from_pretrained(REPO_ID)
policy.config.device = "cuda" if torch.cuda.is_available() else "cpu"
device = get_safe_torch_device(policy.config.device)
policy.to(device).eval()

preprocessor, postprocessor = make_pre_post_processors(policy.config, pretrained_path=REPO_ID)
print(f"[deploy] 로드 완료 | device={device} | 카메라={list(policy.config.image_features)} | "
      f"chunk_size={policy.config.chunk_size} | n_action_steps={policy.config.n_action_steps}")

# ============================================================================
# 2. Scene 구성 (pick_place_collect.py 와 동일한 카메라 배치)
# ============================================================================
world = World(stage_units_in_meters=1.0)
world.scene.add_default_ground_plane()
stage = omni.usd.get_context().get_stage()
franka = world.scene.add(Franka(prim_path="/World/Franka", name="franka"))

cube = world.scene.add(DynamicCuboid(
    prim_path="/World/PickCube", name="pick_cube",
    position=np.array(CUBE_POS, dtype=float), size=0.05,
    color=np.array([0.8, 0.2, 0.1]), mass=0.1))

marker = stage.DefinePrim("/World/PlaceMarker", "Cylinder")
UsdGeom.Cylinder(marker).GetRadiusAttr().Set(0.025)
UsdGeom.Cylinder(marker).GetHeightAttr().Set(0.002)
UsdGeom.XformCommonAPI(marker).SetTranslate(Gf.Vec3d(*TARGET_POS))
UsdGeom.Gprim(marker).GetDisplayColorAttr().Set([Gf.Vec3f(0.1, 0.8, 0.1)])

ee_path = "/World/Franka/panda_hand"
wrist_xform = stage.DefinePrim(ee_path + "/WristCam", "Xform")
wrist_cam_prim = UsdGeom.Camera.Define(stage, WRIST_CAM_PRIM)
wrist_cam_prim.GetFocalLengthAttr().Set(16.0)
wrist_cam_prim.GetHorizontalApertureAttr().Set(20.955)
wrist_cam_prim.GetVerticalApertureAttr().Set(15.716)
wrist_cam_prim.GetClippingRangeAttr().Set(Gf.Vec2f(0.05, 100.0))
UsdGeom.XformCommonAPI(wrist_xform).SetTranslate(Gf.Vec3d(0.15, 0.0, 0.0))
UsdGeom.XformCommonAPI(wrist_xform).SetRotate(Gf.Vec3f(-45, 179.9, -89.9))

over_xform = stage.DefinePrim("/World/OverheadCam", "Xform")
over_cam_prim = UsdGeom.Camera.Define(stage, OVER_CAM_PRIM)
over_cam_prim.GetFocalLengthAttr().Set(24.0)
over_cam_prim.GetHorizontalApertureAttr().Set(20.955)
over_cam_prim.GetVerticalApertureAttr().Set(15.716)
over_cam_prim.GetClippingRangeAttr().Set(Gf.Vec2f(0.05, 100.0))
UsdGeom.XformCommonAPI(over_xform).SetTranslate(Gf.Vec3d(0.4, 0.0, 1.5))
UsdGeom.XformCommonAPI(over_xform).SetRotate(Gf.Vec3f(0.0, 0.0, -89.9))

dome_light = UsdLux.DomeLight.Define(stage, "/World/DR_DomeLight")
dome_light.GetIntensityAttr().Set(1000.0)

for _ in range(50):
    simulation_app.update()

# pick_place_collect.py는 매 에피소드 큐브 색/조명 세기를 랜덤화했다(도메인 랜덤화).
# 여기서는 고정된 빨간 큐브 + 고정 조명(1000)만 썼는데, 이게 학습 분포의 "가장자리"
# 조합일 수 있으므로 학습 때와 동일한 방식으로 한 번 랜덤화한다.
cube_material = cube.get_applied_visual_material()

def randomize_domain():
    cube_material.set_color(np.array([random.uniform(0, 1), random.uniform(0, 1), random.uniform(0, 1)]))
    dome_light.GetIntensityAttr().Set(random.uniform(500.0, 2500.0))

randomize_domain()

# 손목/천장 카메라를 별도 render product로 분리해 동기화 문제를 방지한다.
import omni.replicator.core as rep
rep.create.render_product(WRIST_CAM_PRIM, resolution=(IMG_W, IMG_H))
rep.create.render_product(OVER_CAM_PRIM, resolution=(IMG_W, IMG_H))

wrist_cam = Camera(prim_path=WRIST_CAM_PRIM, resolution=(IMG_W, IMG_H))
over_cam  = Camera(prim_path=OVER_CAM_PRIM,  resolution=(IMG_W, IMG_H))

world.reset()
wrist_cam.initialize()
over_cam.initialize()
for _ in range(WARMUP_STEPS):
    world.step(render=True)

n_dof = franka.num_dof
start_full = np.array(START_POSE, dtype=float)
if len(start_full) < n_dof:
    cur = np.array(franka.get_joint_positions(), dtype=float)
    cur[:len(start_full)] = start_full
    start_full = cur
for _ in range(60):
    franka.apply_action(ArticulationAction(joint_positions=start_full))
    world.step(render=True)
print(f"[deploy] 시작 자세 이동 완료: {np.round(np.array(franka.get_joint_positions())[:7], 3)}")


def grab_rgb(cam):
    """(H,W,3) uint8 RGB 안전 캡처."""
    try:
        rgba = cam.get_rgba()
        if rgba is None or getattr(rgba, "ndim", 0) != 3 or rgba.shape[0] < 2:
            return np.zeros((IMG_H, IMG_W, 3), dtype=np.uint8)
        img = rgba[:, :, :3]
        if img.dtype != np.uint8:
            img = (np.clip(img * 255.0, 0, 255) if float(img.max()) <= 1.0
                   else np.clip(img, 0, 255)).astype(np.uint8)
        return np.ascontiguousarray(img, dtype=np.uint8)
    except Exception:
        return np.zeros((IMG_H, IMG_W, 3), dtype=np.uint8)


def build_observation():
    jp = np.array(franka.get_joint_positions(), dtype=np.float32)[:9]
    jv = np.clip(np.array(franka.get_joint_velocities(), dtype=np.float32), -50.0, 50.0)[:9]
    state = np.concatenate([jp, jv]).astype(np.float32)          # (18,) = qpos+qvel
    obs = {
        "observation.images.wrist": grab_rgb(wrist_cam),
        "observation.images.over":  grab_rgb(over_cam),
        "observation.state":        state,
    }
    return obs, jp

# ============================================================================
# 3. 추론 루프
# ============================================================================
# 정책 내부의 액션 큐가 chunk_size(=n_action_steps)마다 한 번씩만 실제로 추론을
# 수행하고 나머지 스텝은 큐에서 꺼내 쓴다(predict_action을 매 스텝 호출해도 안전).
target_pos = np.array(TARGET_POS, dtype=np.float32)
max_lift = 0.025
success = False
policy.reset()
preprocessor.reset()
postprocessor.reset()

print(f"\n[deploy] ACT(허브) 정책으로 Franka 제어 시작\n")
for step in range(MAX_STEPS):
    obs, jp = build_observation()
    action = predict_action(
        observation=obs, policy=policy, device=device,
        preprocessor=preprocessor, postprocessor=postprocessor,
        use_amp=policy.config.use_amp, task=TASK, robot_type="franka",
    )
    cmd = action.squeeze(0).to("cpu").numpy().astype(float)      # (9,)

    franka.apply_action(ArticulationAction(joint_positions=cmd))
    world.step(render=not HEADLESS)

    gt_cube, _ = cube.get_world_pose()
    gt_cube = np.array(gt_cube, dtype=np.float32)
    max_lift = max(max_lift, float(gt_cube[2]))
    cube_target_xy = float(np.linalg.norm(gt_cube[:2] - target_pos[:2]))
    gripper_open = jp[7] > 0.035

    if step % 60 == 0:
        print(f"[deploy] t={step:4d} | 큐브 {np.round(gt_cube, 3)} | 목표거리 {cube_target_xy:.3f}m | "
              f"최대들림 {max_lift:.3f} | 팔목표 {np.round(cmd[:7], 2)}")

    if cube_target_xy < SUCCESS_XY_TOL and max_lift > (0.025 + SUCCESS_MIN_LIFT) and gripper_open:
        success = True
        print(f"\n[deploy] 성공! 목표오차 {cube_target_xy:.3f}m (최대들림 {max_lift:.3f}m)")
        for _ in range(120):
            world.step(render=not HEADLESS)
        break

print(f"\n[deploy] 종료 | 성공={success} | 최대들림={max_lift:.3f}m")
simulation_app.close()
