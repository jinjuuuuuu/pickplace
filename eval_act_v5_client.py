#!/usr/bin/env python3
# eval_act_v5_client.py
# ---------------------------------------------------------------------------
# Isaac Sim python.sh(3.11)에서 실행. 시뮬 + 카메라만 담당하고, 액션은
# act_policy_server.py(conda lerobot 3.12)에 소켓으로 물어본다. lerobot을
# import하지 않으므로 버전 충돌이 없다.
#
# 실행 (터미널/tmux 창 2개):
#   창1:  conda activate lerobot && python ~/pickplace/act_policy_server.py
#   창2:  /data/isaacsim/python.sh ~/pickplace/eval_act_v5_client.py
#
# 결과: eval_results_act_v5.json (성공률 등). CUBE_XY_LIST는 v4와 동일 → 직접 비교용.
# ---------------------------------------------------------------------------
import json
import socket
import struct
import pickle
import time
import numpy as np

HOST = "127.0.0.1"
PORT = 5555

HEADLESS = True
MAX_STEPS = 1500
IMG_W, IMG_H = 160, 120

TARGET_POS = [0.50, -0.15, 0.025]
START_POSE = [0.0, -0.3, 0.0, -2.5, 0.0, 2.2, 0.8, 0.04, 0.04]

SUCCESS_XY_TOL = 0.05
SUCCESS_MIN_LIFT = 0.04

# v4 평가와 동일한 15개 위치 (seed=123, 타겟서 >=0.15m) → v4 vs v5 직접 비교.
CUBE_XY_LIST = [
    (0.344, 0.156), (0.505, 0.195), (0.485, 0.065), (0.500, 0.009), (0.424, 0.041),
    (0.346, -0.243), (0.418, 0.114), (0.530, 0.063), (0.529, 0.182), (0.355, 0.183),
    (0.499, 0.183), (0.375, 0.014), (0.318, 0.042), (0.359, 0.132), (0.343, -0.094),
]

WRIST_CAM_PRIM = "/World/Franka/panda_hand/WristCam/Camera"
OVER_CAM_PRIM = "/World/OverheadCam/Camera"


# ---- 소켓 헬퍼 (서버와 동일 프로토콜) ----
def send_msg(sock, obj):
    data = pickle.dumps(obj, protocol=4)
    sock.sendall(struct.pack(">I", len(data)) + data)


def recvall(sock, n):
    buf = bytearray()
    while len(buf) < n:
        chunk = sock.recv(n - len(buf))
        if not chunk:
            return None
        buf.extend(chunk)
    return bytes(buf)


def recv_msg(sock):
    raw = recvall(sock, 4)
    if raw is None:
        return None
    n = struct.unpack(">I", raw)[0]
    data = recvall(sock, n)
    if data is None:
        return None
    return pickle.loads(data)


# ---- 정책 서버 연결 (시뮬 켜기 전에 먼저 연결해서, 서버 없으면 빨리 실패) ----
print(f"[client] connecting to policy server {HOST}:{PORT} ...")
sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
connected = False
for attempt in range(60):
    try:
        sock.connect((HOST, PORT))
        connected = True
        break
    except (ConnectionRefusedError, OSError):
        if attempt == 0:
            print("[client] 서버가 아직 안 떴어요. act_policy_server.py 를 먼저 실행하세요. 재시도 중...")
        time.sleep(1.0)
if not connected:
    raise SystemExit("[client] 서버 연결 실패 — act_policy_server.py 가 실행 중인지 확인하세요.")
print("[client] connected to policy server")


def act_remote(wrist_img, over_img, state):
    send_msg(sock, {
        "cmd": "act",
        "wrist": wrist_img.tobytes(),
        "over": over_img.tobytes(),
        "shape": list(wrist_img.shape),
        "state": [float(x) for x in state],
    })
    resp = recv_msg(sock)
    if resp is None or "action" not in resp:
        raise RuntimeError(f"[client] 서버 응답 오류: {resp}")
    return np.asarray(resp["action"], dtype=float)


def reset_remote():
    send_msg(sock, {"cmd": "reset"})
    recv_msg(sock)


# ---- Isaac Sim 시작 (eval_compare_act_v5.py 와 동일한 씬) ----
from isaacsim import SimulationApp
simulation_app = SimulationApp({"renderer": "RayTracedLighting", "headless": HEADLESS})

import omni.usd
from pxr import UsdGeom, Gf, UsdLux
from isaacsim.core.api import World
from isaacsim.core.api.objects import DynamicCuboid
from isaacsim.core.utils.types import ArticulationAction
from isaacsim.robot.manipulators.examples.franka import Franka
from isaacsim.sensors.camera import Camera

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

print(f"[client] starting {len(CUBE_XY_LIST)} episodes")
for ep, (cx, cy) in enumerate(CUBE_XY_LIST):
    world.reset()
    cube.set_world_pose(position=np.array([cx, cy, 0.025], dtype=float))
    for _ in range(10):
        world.step(render=not HEADLESS)
    move_to_start()

    reset_remote()   # 서버 정책 상태 초기화

    max_lift = 0.025
    success = False
    min_xy_err = None
    grasp_attempted = False
    final_step = MAX_STEPS

    for step in range(MAX_STEPS):
        w = grab_rgb(wrist_cam)
        o = grab_rgb(over_cam)
        jp = np.array(franka.get_joint_positions(), dtype=np.float32)[:9]
        cmd = act_remote(w, o, jp)
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
    print(f"[client] ep {ep:2d} cube=({cx:.3f},{cy:.3f}) success={success} steps={final_step} "
          f"max_lift={max_lift:.3f} min_xy_err={min_xy_err:.3f} grasp_attempted={grasp_attempted}")

n_success = sum(r["success"] for r in results)
summary = {
    "model": "ACT v5 (grid-sampled, via policy server)",
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

print(f"[client] DONE success_rate={summary['success_rate']:.2f} ({n_success}/{len(results)}) "
      f"grasp_attempted={summary['n_grasp_attempted']}/{len(results)}")

try:
    send_msg(sock, {"cmd": "bye"})
    recv_msg(sock)
except Exception:
    pass
sock.close()
simulation_app.close()
