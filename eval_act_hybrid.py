# eval_act_hybrid.py - ACT drives approach+grasp; once the gripper is confirmed
# closed near the correct width (~0.025, matching the cube), control is handed off
# to Isaac Sim's built-in PickPlaceController (phase 4 onward: lift/move/lower/release)
# instead of letting ACT try (and fail) to trigger the lift itself.
# Uses the SAME fixed cube list / success criteria as eval_compare_act.py.
import json
import numpy as np

REPO_ID = "jamongsteak/act_pickplace_model"
TASK = "pick up the cube and place it on the target"
HEADLESS = True
ACT_MAX_STEPS = 900          # cap on the ACT-driven approach+grasp phase
HANDOFF_MAX_STEPS = 700      # cap on the scripted lift/move/place phase
IMG_W, IMG_H = 160, 120

TARGET_POS = [0.50, -0.15, 0.025]
START_POSE = [0.0, -0.3, 0.0, -2.5, 0.0, 2.2, 0.8, 0.04, 0.04]
PLACE_Z_OFFSET = 0.02

SUCCESS_XY_TOL = 0.05
SUCCESS_MIN_LIFT = 0.04
GRASP_WIDTH_THRESH = 0.032   # jp[7] must drop below this
GRASP_STABLE_STEPS = 15      # ...for this many consecutive steps before handoff

CUBE_XY_LIST = [
    (0.344, 0.156), (0.505, 0.195), (0.485, 0.065), (0.500, 0.009), (0.424, 0.041),
    (0.346, -0.243), (0.418, 0.114), (0.530, 0.063), (0.529, 0.182), (0.355, 0.183),
    (0.499, 0.183), (0.375, 0.014), (0.318, 0.042), (0.359, 0.132), (0.343, -0.094),
]

WRIST_CAM_PRIM = "/World/Franka/panda_hand/WristCam/Camera"
OVER_CAM_PRIM = "/World/OverheadCam/Camera"

from isaacsim import SimulationApp
simulation_app = SimulationApp({"renderer": "RayTracedLighting", "headless": HEADLESS})

import torch
import omni.usd
from pxr import UsdGeom, Gf, UsdLux
from isaacsim.core.api import World
from isaacsim.core.api.objects import DynamicCuboid
from isaacsim.core.utils.types import ArticulationAction
from isaacsim.robot.manipulators.examples.franka import Franka
from isaacsim.robot.manipulators.examples.franka.controllers import PickPlaceController
from isaacsim.sensors.camera import Camera

from lerobot.policies.act.modeling_act import ACTPolicy
from lerobot.policies.factory import make_pre_post_processors
from lerobot.utils.control_utils import predict_action
from lerobot.utils.utils import get_safe_torch_device

policy = ACTPolicy.from_pretrained(REPO_ID)
policy.config.device = "cuda" if torch.cuda.is_available() else "cpu"
device = get_safe_torch_device(policy.config.device)
policy.to(device).eval()
preprocessor, postprocessor = make_pre_post_processors(policy.config, pretrained_path=REPO_ID)

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
    jp = np.array(franka.get_joint_positions(), dtype=np.float32)[:9]
    jv = np.clip(np.array(franka.get_joint_velocities(), dtype=np.float32), -50.0, 50.0)[:9]
    state = np.concatenate([jp, jv]).astype(np.float32)
    obs = {
        "observation.images.wrist": grab_rgb(wrist_cam),
        "observation.images.over": grab_rgb(over_cam),
        "observation.state": state,
    }
    return obs, jp


def action_to_full(action, current_joint_pos, n_dof):
    full = current_joint_pos.copy()
    if action.joint_positions is None:
        return full
    jpv = np.array(action.joint_positions, dtype=np.float32).flatten()
    if action.joint_indices is not None:
        indices = np.array(action.joint_indices, dtype=int).flatten()
        for i, idx in enumerate(indices):
            if 0 <= idx < n_dof and i < len(jpv):
                full[idx] = jpv[i]
    elif len(jpv) == n_dof:
        full = jpv
    else:
        full[:min(len(jpv), n_dof)] = jpv[:min(len(jpv), n_dof)]
    return full


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
place_target = target_pos.copy()
place_target[2] = target_pos[2] + PLACE_Z_OFFSET
results = []

print(f"[eval_hybrid] starting {len(CUBE_XY_LIST)} episodes")
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
    handed_off = False
    handoff_step = None
    stable_count = 0
    total_steps = 0

    # --- Phase A: ACT drives approach + grasp ---
    for step in range(ACT_MAX_STEPS):
        obs, jp = build_observation()
        action = predict_action(
            observation=obs, policy=policy, device=device,
            preprocessor=preprocessor, postprocessor=postprocessor,
            use_amp=policy.config.use_amp, task=TASK, robot_type="franka",
        )
        cmd = action.squeeze(0).to("cpu").numpy().astype(float)
        franka.apply_action(ArticulationAction(joint_positions=cmd))
        world.step(render=not HEADLESS)
        total_steps += 1

        gt_cube, _ = cube.get_world_pose()
        gt_cube = np.array(gt_cube, dtype=np.float32)
        max_lift = max(max_lift, float(gt_cube[2]))
        cube_target_xy = float(np.linalg.norm(gt_cube[:2] - target_pos[:2]))
        min_xy_err = cube_target_xy if min_xy_err is None else min(min_xy_err, cube_target_xy)
        if jp[7] < 0.035:
            grasp_attempted = True

        if jp[7] < GRASP_WIDTH_THRESH:
            stable_count += 1
        else:
            stable_count = 0

        if stable_count >= GRASP_STABLE_STEPS:
            handed_off = True
            handoff_step = step
            break

    # --- Phase B: hand off to the scripted PickPlaceController (lift onward) ---
    if handed_off:
        gt_cube, _ = cube.get_world_pose()
        cube_pos_now = np.array(gt_cube, dtype=np.float32)
        controller = PickPlaceController(name=f"hybrid_ctrl_{ep}", gripper=franka.gripper, robot_articulation=franka)
        controller._event = 4
        controller._t = 0.0
        controller._h0 = float(cube_pos_now[2])
        controller._current_target_x = float(cube_pos_now[0])
        controller._current_target_y = float(cube_pos_now[1])

        for step in range(HANDOFF_MAX_STEPS):
            current_joint_pos = np.array(franka.get_joint_positions(), dtype=np.float32)
            action = controller.forward(
                picking_position=cube_pos_now, placing_position=place_target,
                current_joint_positions=current_joint_pos,
            )
            full_action = action_to_full(action, current_joint_pos, n_dof)
            franka.apply_action(ArticulationAction(joint_positions=full_action))
            world.step(render=not HEADLESS)
            total_steps += 1

            gt_cube, _ = cube.get_world_pose()
            gt_cube = np.array(gt_cube, dtype=np.float32)
            max_lift = max(max_lift, float(gt_cube[2]))
            cube_target_xy = float(np.linalg.norm(gt_cube[:2] - target_pos[:2]))
            min_xy_err = min(min_xy_err, cube_target_xy)
            jp7 = float(current_joint_pos[7])
            gripper_open = jp7 > 0.035

            if cube_target_xy < SUCCESS_XY_TOL and max_lift > (0.025 + SUCCESS_MIN_LIFT) and gripper_open:
                success = True
                break

            if controller.is_done():
                break

    results.append({
        "episode": ep, "cube_xy": [cx, cy], "success": success, "steps": total_steps,
        "max_lift": round(max_lift, 4), "min_xy_err": round(float(min_xy_err), 4),
        "grasp_attempted": grasp_attempted, "handed_off": handed_off, "handoff_step": handoff_step,
    })
    print(f"[eval_hybrid] ep {ep:2d} cube=({cx:.3f},{cy:.3f}) success={success} handed_off={handed_off} "
          f"handoff_step={handoff_step} max_lift={max_lift:.3f} min_xy_err={min_xy_err:.3f}")

n_success = sum(r["success"] for r in results)
n_handoff = sum(r["handed_off"] for r in results)
summary = {
    "model": "ACT (approach+grasp) + scripted PickPlaceController handoff (lift/move/place)",
    "n_episodes": len(results),
    "n_success": n_success,
    "success_rate": n_success / len(results),
    "n_handed_off": n_handoff,
    "avg_min_xy_err": float(np.mean([r["min_xy_err"] for r in results])),
    "episodes": results,
}
with open("eval_results_act_hybrid.json", "w") as f:
    json.dump(summary, f, indent=2)

print(f"[eval_hybrid] DONE success_rate={summary['success_rate']:.2f} ({n_success}/{len(results)}) "
      f"handed_off={n_handoff}/{len(results)}")
simulation_app.close()
