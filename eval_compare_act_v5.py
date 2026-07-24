# eval_compare_act_v5.py - 격자 샘플링(v5)으로 학습한 ACT 정책의 새 위치 일반화 평가.
# ---------------------------------------------------------------------------
# eval_compare_act_v4.py 를 기반으로, 아래만 바꿈:
#   - 모델을 HF 허브가 아니라 '로컬 체크포인트'에서 로드 (lerobot-train 결과)
#   - 결과를 eval_results_act_v5.json 에 저장, 라벨 v5
# CUBE_XY_LIST(15개 위치)는 v4와 '동일'하게 유지 → v4(연속 샘플링) vs
# v5(격자 샘플링) 성공률을 같은 시나리오에서 직접 비교하기 위함.
#
# 실행 (워크스테이션, Isaac Sim python.sh):
#   /data/isaacsim/python.sh ~/pickplace/eval_compare_act_v5.py
# ---------------------------------------------------------------------------
import os
import json
import numpy as np

# 🔧 lerobot-train --output_dir=/data/jinju/act_pickplace_v5 의 결과 체크포인트 경로.
#    실제 경로는  ls /data/jinju/act_pickplace_v5/checkpoints/  로 확인 후 맞출 것.
#    보통 'last'(마지막 스텝 심볼릭 링크) 아래 pretrained_model 폴더가 있다.
MODEL_PATH = "/data/jinju/act_pickplace_v5/checkpoints/last/pretrained_model"

TASK = "pick up the cube and place it on the target"
HEADLESS = True
MAX_STEPS = 1500
IMG_W, IMG_H = 160, 120

TARGET_POS = [0.50, -0.15, 0.025]
START_POSE = [0.0, -0.3, 0.0, -2.5, 0.0, 2.2, 0.8, 0.04, 0.04]

SUCCESS_XY_TOL = 0.05
SUCCESS_MIN_LIFT = 0.04

# Fixed cube (x,y) list, generated once with seed=123, all >=0.15m from TARGET_POS.
# (v4 평가와 동일 목록 → 직접 비교용)
CUBE_XY_LIST = [
    (0.344, 0.156), (0.505, 0.195), (0.485, 0.065), (0.500, 0.009), (0.424, 0.041),
    (0.346, -0.243), (0.418, 0.114), (0.530, 0.063), (0.529, 0.182), (0.355, 0.183),
    (0.499, 0.183), (0.375, 0.014), (0.318, 0.042), (0.359, 0.132), (0.343, -0.094),
]

WRIST_CAM_PRIM = "/World/Franka/panda_hand/WristCam/Camera"
OVER_CAM_PRIM = "/World/OverheadCam/Camera"

# 모델 경로 사전 확인 (오타/미학습 시 친절히 안내)
if not os.path.isdir(MODEL_PATH):
    raise SystemExit(
        f"[eval_act_v5] 체크포인트 폴더가 없습니다: {MODEL_PATH}\n"
        f"  ls /data/jinju/act_pickplace_v5/checkpoints/  로 실제 경로를 확인하고 "
        f"이 파일 상단의 MODEL_PATH를 맞춰주세요.")

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

print(f"[eval_act_v5] loading policy from local checkpoint: {MODEL_PATH}")
policy = ACTPolicy.from_pretrained(MODEL_PATH)
policy.config.device = "cuda" if torch.cuda.is_available() else "cpu"
device = get_safe_torch_device(policy.config.device)
policy.to(device).eval()
preprocessor, postprocessor = make_pre_post_processors(policy.config, pretrained_path=MODEL_PATH)
print("[eval_act_v5] policy loaded, device=", device)

world = World(stage_units_in_meters=1.0)
world.scene.add_default_ground_plane()
stage = omni.usd.get_context().get_stage()
franka = world.scene.add(Franka(prim_path="/World/Franka", name="franka"))

cube = world.scene.add(DynamicCuboid(
    prim_path="/World/PickCube", name="pick_cube",
    position=np.array([CUBE_XY_LIST[0][0], CUBE_XY_LIST[0][1], 0.025], dtype=float), size=0.05,
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

import omni.replicator.core as rep
rep.create.render_product(WRIST_CAM_PRIM, resolution=(IMG_W, IMG_H))
rep.create.render_product(OVER_CAM_PRIM, resolution=(IMG_W, IMG_H))

wrist_cam = Camera(prim_path=WRIST_CAM_PRIM, resolution=(IMG_W, IMG_H))
over_cam = Camera(prim_path=OVER_CAM_PRIM, resolution=(IMG_W, IMG_H))

world.reset()
wrist_cam.initialize()
over_cam.initialize()
for _ in range(15):
    world.step(render=not HEADLESS)

n_dof = franka.num_dof


def grab_rgb(cam):
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
    # v5 dataset stores joint_pos (9) only as state (convert_to_lerobot.py PROPRIO_DIM=9).
    jp = np.array(franka.get_joint_positions(), dtype=np.float32)[:9]
    state = jp.astype(np.float32)
    obs = {
        "observation.images.wrist": grab_rgb(wrist_cam),
        "observation.images.over": grab_rgb(over_cam),
        "observation.state": state,
    }
    return obs, jp


def move_to_start():
    start_full = np.array(START_POSE, dtype=float)
    if len(start_full) < n_dof:
        cur = np.array(franka.get_joint_positions(), dtype=float)
        cur[:len(start_full)] = start_full
        start_full = cur
    for _ in range(60):
        franka.apply_action(ArticulationAction(joint_positions=start_full))
        world.step(render=not HEADLESS)


target_pos = np.array(TARGET_POS, dtype=np.float32)
results = []

print(f"[eval_act_v5] starting {len(CUBE_XY_LIST)} episodes")
for ep, (cx, cy) in enumerate(CUBE_XY_LIST):
    world.reset()
    cube.set_world_pose(position=np.array([cx, cy, 0.025], dtype=float))
    for _ in range(10):
        world.step(render=not HEADLESS)
    move_to_start()

    policy.reset()
    preprocessor.reset()
    postprocessor.reset()

    max_lift = 0.025
    success = False
    min_xy_err = None
    grasp_attempted = False
    final_step = MAX_STEPS

    for step in range(MAX_STEPS):
        obs, jp = build_observation()
        action = predict_action(
            observation=obs, policy=policy, device=device,
            preprocessor=preprocessor, postprocessor=postprocessor,
            use_amp=policy.config.use_amp, task=TASK, robot_type="franka",
        )
        cmd = action.squeeze(0).to("cpu").numpy().astype(float)
        franka.apply_action(ArticulationAction(joint_positions=cmd))
        world.step(render=not HEADLESS)

        gt_cube, _ = cube.get_world_pose()
        gt_cube = np.array(gt_cube, dtype=np.float32)
        max_lift = max(max_lift, float(gt_cube[2]))
        cube_target_xy = float(np.linalg.norm(gt_cube[:2] - target_pos[:2]))
        min_xy_err = cube_target_xy if min_xy_err is None else min(min_xy_err, cube_target_xy)
        if jp[7] < 0.035:
            grasp_attempted = True
        gripper_open = jp[7] > 0.035

        if cube_target_xy < SUCCESS_XY_TOL and max_lift > (0.025 + SUCCESS_MIN_LIFT) and gripper_open:
            success = True
            final_step = step
            break

    results.append({
        "episode": ep, "cube_xy": [cx, cy], "success": success, "steps": final_step,
        "max_lift": round(max_lift, 4), "min_xy_err": round(float(min_xy_err), 4),
        "grasp_attempted": grasp_attempted,
    })
    print(f"[eval_act_v5] ep {ep:2d} cube=({cx:.3f},{cy:.3f}) success={success} steps={final_step} "
          f"max_lift={max_lift:.3f} min_xy_err={min_xy_err:.3f} grasp_attempted={grasp_attempted}")

n_success = sum(r["success"] for r in results)
summary = {
    "model": f"ACT v5 (local grid-sampled, {MODEL_PATH})",
    "n_episodes": len(results),
    "n_success": n_success,
    "success_rate": n_success / len(results),
    "avg_steps_success": float(np.mean([r["steps"] for r in results if r["success"]])) if n_success else None,
    "avg_min_xy_err": float(np.mean([r["min_xy_err"] for r in results])),
    "n_grasp_attempted": sum(r["grasp_attempted"] for r in results),
    "episodes": results,
}
with open("eval_results_act_v5.json", "w") as f:
    json.dump(summary, f, indent=2)

print(f"[eval_act_v5] DONE success_rate={summary['success_rate']:.2f} ({n_success}/{len(results)}) "
      f"grasp_attempted={summary['n_grasp_attempted']}/{len(results)}")
simulation_app.close()
