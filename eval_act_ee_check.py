# eval_act_ee_check.py - single episode, logs end-effector XY vs cube XY around the
# grasp window to check whether ACT actually closes the gripper AT the cube, or
# somewhere offset from it (closing on empty air).
import numpy as np

REPO_ID = "jamongsteak/act_pickplace_model"
TASK = "pick up the cube and place it on the target"
HEADLESS = True
MAX_STEPS = 500
IMG_W, IMG_H = 160, 120
CUBE_XY = (0.500, 0.009)  # the novel/unseen position that previously failed to converge
TARGET_POS = [0.50, -0.15, 0.025]
START_POSE = [0.0, -0.3, 0.0, -2.5, 0.0, 2.2, 0.8, 0.04, 0.04]
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
from isaacsim.sensors.camera import Camera

from lerobot.policies.act.modeling_act import ACTPolicy
from lerobot.policies.factory import make_pre_post_processors
from lerobot.utils.control_utils import predict_action
from lerobot.utils.utils import get_safe_torch_device

policy = ACTPolicy.from_pretrained(REPO_ID)
policy.config.device = "cuda" if torch.cuda.is_available() else "cpu"
policy.config.n_action_steps = 20  # was 100 (== chunk_size); replan more often for closed-loop correction
device = get_safe_torch_device(policy.config.device)
policy.to(device).eval()
preprocessor, postprocessor = make_pre_post_processors(policy.config, pretrained_path=REPO_ID)

world = World(stage_units_in_meters=1.0)
world.scene.add_default_ground_plane()
stage = omni.usd.get_context().get_stage()
franka = world.scene.add(Franka(prim_path="/World/Franka", name="franka"))
cube = world.scene.add(DynamicCuboid(
    prim_path="/World/PickCube", name="pick_cube",
    position=np.array([CUBE_XY[0], CUBE_XY[1], 0.025], dtype=float), size=0.05,
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
wrist_cam.initialize(); over_cam.initialize()
for _ in range(15):
    world.step(render=not HEADLESS)

n_dof = franka.num_dof
start_full = np.array(START_POSE, dtype=float)
for _ in range(60):
    franka.apply_action(ArticulationAction(joint_positions=start_full))
    world.step(render=not HEADLESS)


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


def get_ee_xy():
    try:
        ee_pos, _ = franka.end_effector.get_world_pose()
        return np.array(ee_pos[:2], dtype=np.float32)
    except Exception:
        stage2 = omni.usd.get_context().get_stage()
        hand = stage2.GetPrimAtPath("/World/Franka/panda_hand")
        xf = UsdGeom.Xformable(hand).ComputeLocalToWorldTransform(0)
        t = xf.ExtractTranslation()
        return np.array([t[0], t[1]], dtype=np.float32)


policy.reset(); preprocessor.reset(); postprocessor.reset()
target_pos = np.array(TARGET_POS, dtype=np.float32)
log_lines = []
for step in range(MAX_STEPS):
    jp = np.array(franka.get_joint_positions(), dtype=np.float32)[:9]
    jv = np.clip(np.array(franka.get_joint_velocities(), dtype=np.float32), -50.0, 50.0)[:9]
    state = np.concatenate([jp, jv]).astype(np.float32)
    obs = {
        "observation.images.wrist": grab_rgb(wrist_cam),
        "observation.images.over": grab_rgb(over_cam),
        "observation.state": state,
    }
    action = predict_action(
        observation=obs, policy=policy, device=device,
        preprocessor=preprocessor, postprocessor=postprocessor,
        use_amp=policy.config.use_amp, task=TASK, robot_type="franka",
    )
    cmd = action.squeeze(0).to("cpu").numpy().astype(float)
    franka.apply_action(ArticulationAction(joint_positions=cmd))
    world.step(render=not HEADLESS)

    gt_cube, _ = cube.get_world_pose()
    cube_xy = np.array(gt_cube[:2], dtype=np.float32)
    ee_xy = get_ee_xy()
    ee_cube_dist = float(np.linalg.norm(ee_xy - cube_xy))
    if step % 20 == 0 or (jp[7] < 0.037):
        log_lines.append(
            f"t={step:4d} jp7={jp[7]:.4f} ee_xy=({ee_xy[0]:.3f},{ee_xy[1]:.3f}) "
            f"cube_xy=({cube_xy[0]:.3f},{cube_xy[1]:.3f}) ee_cube_xy_dist={ee_cube_dist:.4f}"
        )

with open("act_ee_check_log_n20.txt", "w") as f:
    f.write("\n".join(log_lines))

simulation_app.close()
