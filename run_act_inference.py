# run_act_inference.py
import os
import random
import numpy as np
import torch

# === 설정 ===
NUM_EPISODES = 10
MAX_STEPS    = 2000
REPO_ID      = "jamongsteak/act_pickplace_model" # 내 허깅페이스 모델
IMG_W, IMG_H = 160, 120
# pick_place_collect.py(학습 데이터 수집 스크립트)와 반드시 동일해야 하는 값들.
START_POSE   = [0.0, -0.3, 0.0, -2.5, 0.0, 2.2, 0.8, 0.04, 0.04]
START_SETTLE_STEPS = 60

# 1. Isaac Sim 시뮬레이터 켜기
from isaacsim import SimulationApp
simulation_app = SimulationApp({"renderer": "RayTracedLighting", "headless": False})

import omni.usd
from pxr import UsdGeom, Gf
from isaacsim.core.api import World
from isaacsim.core.api.objects import DynamicCuboid
from isaacsim.core.utils.types import ArticulationAction
from isaacsim.robot.manipulators.examples.franka import Franka
from isaacsim.sensors.camera import Camera

# 2. LeRobot ACT 모델 불러오기
from lerobot.policies.act.modeling_act import ACTPolicy
from lerobot.policies.factory import make_pre_post_processors
from lerobot.utils.control_utils import predict_action
from lerobot.utils.utils import get_safe_torch_device

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"[{REPO_ID}] 모델을 다운로드하고 뇌를 이식하는 중입니다...")
policy = ACTPolicy.from_pretrained(REPO_ID)
policy.config.device = device
device = get_safe_torch_device(policy.config.device)
policy.to(device)
policy.eval()

# 학습 때 사용한 정규화 통계(dataset_stats)를 반드시 같이 불러와야 한다.
# select_action()은 정규화되지 않은 값을 그대로 받아 정규화된 값을 그대로 반환하므로,
# 이 전/후처리 파이프라인이 없으면 관절 목표값이 완전히 다른 스케일이 되어 팔이 미친듯이 움직인다.
preprocessor, postprocessor = make_pre_post_processors(policy.config, pretrained_path=REPO_ID)
print("뇌 이식 완료! 시뮬레이션을 구성합니다.")

# 3. 환경 세팅
world = World(stage_units_in_meters=1.0)
world.scene.add_default_ground_plane()
stage = omni.usd.get_context().get_stage()

franka = world.scene.add(Franka(prim_path="/World/Franka", name="franka"))
cube = world.scene.add(DynamicCuboid(
    prim_path="/World/PickCube", name="pick_cube",
    position=np.array([0.45, 0.0, 0.025]), size=0.05,
    color=np.array([0.8, 0.2, 0.1]), mass=0.1
))

# ============================================================================
# 4. 카메라 프리미티브 세팅 (학습 데이터를 만든 pick_place_collect.py와 동일해야 함)
# ============================================================================
WRIST_CAM_PRIM = "/World/Franka/panda_hand/WristCam/Camera"
OVER_CAM_PRIM  = "/World/OverheadCam/Camera"

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
UsdGeom.XformCommonAPI(ox).SetTranslate(Gf.Vec3d(0.4, 0.0, 1.5))
UsdGeom.XformCommonAPI(ox).SetRotate(Gf.Vec3f(0.0, 0.0, -89.9))

# 센서 객체 래핑
wrist_cam = Camera(prim_path=WRIST_CAM_PRIM, resolution=(IMG_W, IMG_H))
over_cam  = Camera(prim_path=OVER_CAM_PRIM,  resolution=(IMG_W, IMG_H))

# 🔥 크래시 방지 핵심: Replicator 코드를 완전히 빼고 world.reset() 후 바로 초기화
world.reset()
wrist_cam.initialize()
over_cam.initialize()
for _ in range(15):
    world.step(render=True)

# 이미지 캡처 함수 (bc_deploy_vision.py의 안정적인 예외 처리 방식 적용)
def grab_rgb(cam):
    try:
        rgba = cam.get_rgba()
        if rgba is None or getattr(rgba, "ndim", 0) != 3 or rgba.shape[0] < 2:
            return np.zeros((IMG_H, IMG_W, 3), dtype=np.uint8)
        img = rgba[:, :, :3]
        if img.dtype != np.uint8:
            img = (np.clip(img * 255.0, 0, 255) if float(img.max()) <= 1.0 else np.clip(img, 0, 255)).astype(np.uint8)
        return np.ascontiguousarray(img, dtype=np.uint8)
    except Exception:
        return np.zeros((IMG_H, IMG_W, 3), dtype=np.uint8)

# ============================================================================
# 5. AI 조종 테스트 루프
# ============================================================================
print("=== AI 조종 테스트 시작 ===")

num_dof = franka.num_dof
finger_idx = np.array([num_dof - 2, num_dof - 1], dtype=int)

for ep in range(NUM_EPISODES):
    world.reset()
    cube.set_world_pose(position=np.array([random.uniform(0.35, 0.5), random.uniform(-0.15, 0.15), 0.025]))

    for _ in range(30):
        world.step(render=True)

    # 학습 데이터는 매 에피소드 이 START_POSE에서 시작한다. 여기서 정렬을 안 해주면
    # Isaac Sim 기본 리셋 자세(학습 때 한 번도 본 적 없는 관절값)에서 첫 관측이 시작되고,
    # ACT는 그 첫 관측으로 100스텝짜리 액션 청크를 예측하므로 처음부터 어긋난 궤적을 그린다.
    start_full = np.array(franka.get_joint_positions(), dtype=float)
    start_full[:7] = np.array(START_POSE[:7], dtype=float)
    start_full[finger_idx] = START_POSE[7]
    for _ in range(START_SETTLE_STEPS):
        franka.apply_action(ArticulationAction(joint_positions=start_full))
        world.step(render=True)

    # 에피소드마다 정책의 액션 큐/전후처리 파이프라인 내부 상태를 초기화해야 한다.
    # 안 하면 이전 에피소드에서 남은(리셋 전 장면 기준으로 예측된) 액션이 큐에 그대로 남아
    # 새 에피소드 초반 몇 스텝 동안 팔이 엉뚱하게 움직인다.
    policy.reset()
    preprocessor.reset()
    postprocessor.reset()

    step = 0
    while step < MAX_STEPS:
        w_img = grab_rgb(wrist_cam)
        o_img = grab_rgb(over_cam)

        joint_pos = np.array(franka.get_joint_positions(), dtype=np.float32)
        joint_vel = np.array(franka.get_joint_velocities(), dtype=np.float32)
        state_18 = np.concatenate([joint_pos, joint_vel])

        obs_dict = {
            "observation.images.wrist": w_img,
            "observation.images.over":  o_img,
            "observation.state":        state_18,
        }

        # predict_action이 정규화(preprocessor) -> policy.select_action -> 역정규화(postprocessor)를
        # 전부 처리해준다. select_action을 직접 부르면 학습 때 쓴 평균/표준편차로
        # 정규화도, 역정규화도 되지 않아 관절 목표값이 전혀 다른 스케일로 나온다.
        action = predict_action(
            observation=obs_dict, policy=policy, device=device,
            preprocessor=preprocessor, postprocessor=postprocessor,
            use_amp=policy.config.use_amp, task="pick up the cube and place it on the target",
            robot_type="franka",
        )

        action_np = action.squeeze(0).cpu().numpy().astype(float)
        franka.apply_action(ArticulationAction(joint_positions=action_np))

        world.step(render=True)
        step += 1

simulation_app.close()