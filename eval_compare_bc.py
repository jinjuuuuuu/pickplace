# eval_compare_bc.py - multi-episode headless success-rate eval for the vision BC
# (frame-stacked ResNet18) policy. Uses the SAME fixed cube position list and
# success criteria as eval_compare_act.py for a fair comparison.
import json
import collections
import numpy as np
import torch
import torch.nn as nn

HEADLESS = True
POLICY_PATH = r"C:\Users\user\Desktop\claude_jetbot\bc_policy_vision_resnet_stacked.pt"
NORM_STATS_PATH = r"C:\Users\user\Desktop\claude_jetbot\bc_policy_vision_resnet_stacked_norm_stats.pt"

MAX_STEPS = 1500
REPLAN_EVERY = 60
TARGET_POS = [0.50, -0.15, 0.025]
SUCCESS_XY_TOL = 0.05
SUCCESS_MIN_LIFT = 0.04
WRIST_CAM_PRIM = "/World/Franka/panda_hand/WristCam/Camera"
OVER_CAM_PRIM = "/World/OverheadCam/Camera"
CAP_W, CAP_H = 160, 120

CUBE_XY_LIST = [
    (0.344, 0.156), (0.505, 0.195), (0.485, 0.065), (0.500, 0.009), (0.424, 0.041),
    (0.346, -0.243), (0.418, 0.114), (0.530, 0.063), (0.529, 0.182), (0.355, 0.183),
    (0.499, 0.183), (0.375, 0.014), (0.318, 0.042), (0.359, 0.132), (0.343, -0.094),
]

from isaacsim import SimulationApp
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
        self.cnn_over = ResNet18Backbone(in_channels=3 * obs_seq, out_dim=feat)
        self.head = nn.Sequential(
            nn.Linear(feat * 2 + proprio_dim * obs_seq, 512), nn.ReLU(),
            nn.Linear(512, 512), nn.ReLU(),
            nn.Linear(512, 256), nn.ReLU(),
            nn.Linear(256, action_dim),
        )

    def forward(self, wrist_img, over_img, proprio):
        fw = self.cnn_wrist(wrist_img)
        fo = self.cnn_over(over_img)
        return self.head(torch.cat([fw, fo, proprio], dim=1))


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[eval_bc] loading policy on {device}")
ckpt = torch.load(POLICY_PATH, map_location=device, weights_only=False)

PROPRIO_DIM = 18
CHUNK_H = 60
JOINT_DIM = 9
ACTION_DIM = CHUNK_H * JOINT_DIM
IMG_SIZE = 84
OBS_SEQ = 3

policy = VisionBCPolicy(PROPRIO_DIM, ACTION_DIM, obs_seq=OBS_SEQ).to(device)
policy.load_state_dict(ckpt["policy"])
policy.eval()

norm = torch.load(NORM_STATS_PATH, map_location="cpu", weights_only=False)
pro_mean = norm["proprio_mean"].numpy()
pro_std = norm["proprio_std"].numpy()
act_mean = norm["act_mean"].numpy()
act_std = norm["act_std"].numpy()
START_POSE = norm["start_pose"].numpy() if "start_pose" in norm else \
    np.array([0.0, -0.3, 0.0, -2.5, 0.0, 2.2, 0.8, 0.04, 0.04], dtype=np.float32)
print(f"[eval_bc] loaded epoch={ckpt['epoch']} val_loss={ckpt.get('val_loss')}")

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

_dome = UsdLux.DomeLight.Define(stage, "/World/DR_DomeLight")
_dome.GetIntensityAttr().Set(1000.0)

wrist_cam = Camera(prim_path=WRIST_CAM_PRIM, resolution=(CAP_W, CAP_H))
over_cam = Camera(prim_path=OVER_CAM_PRIM, resolution=(CAP_W, CAP_H))

world.reset()
wrist_cam.initialize()
over_cam.initialize()
for _ in range(15):
    world.step(render=not HEADLESS)

num_dof = len(franka.get_joint_positions())
n_dof = num_dof


def grab(cam):
    try:
        rgba = cam.get_rgba()
        if rgba is None or getattr(rgba, "ndim", 0) != 3 or rgba.shape[0] < 2:
            return np.zeros((IMG_SIZE, IMG_SIZE, 3), np.uint8)
        img = _resize(rgba[:, :, :3].astype(np.uint8), IMG_SIZE)
        return np.ascontiguousarray(img, dtype=np.uint8)
    except Exception:
        return np.zeros((IMG_SIZE, IMG_SIZE, 3), np.uint8)


def build_inputs():
    jp = np.array(franka.get_joint_positions(), dtype=np.float32)
    jv = np.clip(franka.get_joint_velocities(), -50.0, 50.0).astype(np.float32)
    proprio = np.concatenate([jp, jv])[:PROPRIO_DIM]
    w = grab(wrist_cam)
    o = grab(over_cam)
    return proprio, w, o, jp


def predict_chunk(stacked_wrist, stacked_over, stacked_pro):
    with torch.no_grad():
        out = policy(stacked_wrist, stacked_over, stacked_pro).squeeze(0).cpu().numpy()
    chunk = out.reshape(CHUNK_H, JOINT_DIM) * act_std + act_mean
    for h in range(CHUNK_H):
        chunk[h, 7:9] = 0.0 if chunk[h, 7] < 0.035 else 0.04
    return chunk


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

print(f"[eval_bc] starting {len(CUBE_XY_LIST)} episodes")
for ep, (cx, cy) in enumerate(CUBE_XY_LIST):
    world.reset()
    cube.set_world_pose(position=np.array([cx, cy, 0.025], dtype=float))
    for _ in range(10):
        world.step(render=not HEADLESS)
    move_to_start()

    wrist_queue = collections.deque(maxlen=OBS_SEQ)
    over_queue = collections.deque(maxlen=OBS_SEQ)
    pro_queue = collections.deque(maxlen=OBS_SEQ)
    chunk = None

    max_lift = 0.025
    success = False
    min_xy_err = None
    grasp_attempted = False
    final_step = MAX_STEPS

    for step in range(MAX_STEPS):
        proprio, w, o, jp = build_inputs()
        pn = (proprio - pro_mean) / pro_std
        wt = torch.FloatTensor(w.astype(np.float32).transpose(2, 0, 1) / 255.0).unsqueeze(0)
        ot = torch.FloatTensor(o.astype(np.float32).transpose(2, 0, 1) / 255.0).unsqueeze(0)
        pt = torch.FloatTensor(pn).unsqueeze(0)

        if len(wrist_queue) == 0:
            for _ in range(OBS_SEQ):
                wrist_queue.append(wt)
                over_queue.append(ot)
                pro_queue.append(pt)
        else:
            wrist_queue.append(wt)
            over_queue.append(ot)
            pro_queue.append(pt)

        if chunk is None or step % REPLAN_EVERY == 0:
            stacked_wrist = torch.cat(list(wrist_queue), dim=1).to(device)
            stacked_over = torch.cat(list(over_queue), dim=1).to(device)
            stacked_pro = torch.cat(list(pro_queue), dim=1).to(device)
            chunk = predict_chunk(stacked_wrist, stacked_over, stacked_pro)

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

        h = step % REPLAN_EVERY
        franka.apply_action(ArticulationAction(joint_positions=chunk[h].astype(float)))
        world.step(render=not HEADLESS)

    results.append({
        "episode": ep, "cube_xy": [cx, cy], "success": success, "steps": final_step,
        "max_lift": round(max_lift, 4), "min_xy_err": round(float(min_xy_err), 4),
        "grasp_attempted": grasp_attempted,
    })
    print(f"[eval_bc] ep {ep:2d} cube=({cx:.3f},{cy:.3f}) success={success} steps={final_step} "
          f"max_lift={max_lift:.3f} min_xy_err={min_xy_err:.3f} grasp_attempted={grasp_attempted}")

n_success = sum(r["success"] for r in results)
summary = {
    "model": "BC vision (ResNet18, frame-stacked x3)",
    "n_episodes": len(results),
    "n_success": n_success,
    "success_rate": n_success / len(results),
    "avg_steps_success": float(np.mean([r["steps"] for r in results if r["success"]])) if n_success else None,
    "avg_min_xy_err": float(np.mean([r["min_xy_err"] for r in results])),
    "n_grasp_attempted": sum(r["grasp_attempted"] for r in results),
    "episodes": results,
}
with open("eval_results_bc.json", "w") as f:
    json.dump(summary, f, indent=2)

print(f"[eval_bc] DONE success_rate={summary['success_rate']:.2f} ({n_success}/{len(results)}) "
      f"grasp_attempted={summary['n_grasp_attempted']}/{len(results)}")
simulation_app.close()
