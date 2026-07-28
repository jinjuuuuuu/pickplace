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
# 결과: eval_results_act_v5.json (성공률, 파지 기하, 검은프레임 카운터 등).
# 씬 설정과 평가 격자는 scene_config.py에서 온다 — 수집 스크립트와 같은 파일을
# 읽으므로 두 씬이 어긋날 수 없다(어긋남이 v5~v10 실패의 주원인이었다).
# ---------------------------------------------------------------------------
import os
import json
import socket
import struct
import pickle
import time
import numpy as np

# 씬 설정(카메라·해상도·조명·영역·평가 격자·솎기 stride)은 scene_config.py 한
# 곳에서만 온다. 수집 스크립트도 같은 파일을 읽으므로 두 씬이 어긋날 수 없다.
from scene_config import (
    IMG_W, IMG_H, WRIST_CAM_PRIM, OVER_CAM_PRIM,
    CAM_H_APERTURE, CAM_V_APERTURE, CAM_CLIP_NEAR, CAM_CLIP_FAR,
    WRIST_FOCAL, WRIST_TRANSLATE, WRIST_ROTATE,
    OVER_FOCAL, OVER_TRANSLATE, OVER_ROTATE,
    LIGHT_INTENSITY, CUBE_COLOR, CUBE_SIZE, CUBE_MASS, CUBE_Z,
    TARGET_FIXED_XY, TARGET_Z, START_POSE, TRAIN_STRIDE,
    SUCCESS_XY_TOL, SUCCESS_MIN_LIFT, EVAL_CUBE_XY_LIST,
    GRIPPER_CLOSED, GRIPPER_OPEN, GRIPPER_CLOSING_RAW_THRESH,
    GRIPPER_CLOSE_THRESH as GRIPPER_CLOSE_THRESH_DEFAULT,
)

HOST = "127.0.0.1"
PORT = 5555

HEADLESS = True
# ⚠ 렌더는 headless와 별개로 반드시 True. 예전엔 world.step(render=RENDER)
# 였는데, headless에서 render=False면 Camera.get_rgba()가 빈 배열을 돌려줘서
# grab_rgb()가 온통 검은 이미지를 정책에 먹였다(=무조건 0%). 수집 스크립트는
# RENDER=True로 돌았으니 평가도 같아야 한다. 아래 blank 카운터로 재발을 감시한다.
RENDER = True
MAX_STEPS = 1500

# 학습 데이터를 몇 프레임마다 솎았는지(subsample_dataset.py --stride).
# 물리는 60Hz로 도는데 정책은 (60/stride)Hz로 판단하도록 학습됐으므로, 액션
# 하나를 stride번 유지해야 로봇이 학습 때와 같은 속도로 움직인다. 이 값이
# 틀리면 에러 없이 그냥 실패한다 — 그래서 scene_config.TRAIN_STRIDE를 기본값으로
# 쓴다(예전엔 기본값이 1이라 환경변수를 잊으면 조용히 3배 빠르게 움직였다).
# 솎지 않은 옛 모델(v5 등)을 평가할 때만 ACTION_REPEAT=1로 덮어쓸 것.
ACTION_REPEAT = max(1, int(os.environ.get("ACTION_REPEAT", str(TRAIN_STRIDE))))

# 학습 데이터의 그리퍼 값은 두 가지뿐이다(수집 스크립트에서 강제 이진화).
# ACT는 회귀 모델이라 그 사이 값을 내놓는데, 큐브가 5cm라 손가락이 조금만
# 벌어져도 못 잡는다. 그래서 수집 때와 동일하게 이진화해서 명령한다.
# 값은 scene_config에서 온다 — 라벨과 명령이 어긋난 것이 v10을 0%로 만든 원인이다.
#   GRIPPER_BINARIZE=0        이진화 끄기 (정책 원본값 그대로 명령)
#   GRIPPER_CLOSE_THRESH=...  닫힘 판정 임계값 덮어쓰기
#   GRIPPER_CLOSED_CMD=...    닫을 때 명령하는 값 덮어쓰기
#
# 라벨이 0.025였던 옛 모델(v10 이전)을 평가할 때는 아래처럼 덮어쓸 것:
#   GRIPPER_CLOSE_THRESH=0.0325 GRIPPER_CLOSED_CMD=0.0 ...
GRIPPER_BINARIZE = os.environ.get("GRIPPER_BINARIZE", "1") != "0"
GRIPPER_CLOSE_THRESH = float(os.environ.get(
    "GRIPPER_CLOSE_THRESH", str(GRIPPER_CLOSE_THRESH_DEFAULT)))
GRIPPER_CLOSED_CMD = float(os.environ.get("GRIPPER_CLOSED_CMD", str(GRIPPER_CLOSED)))


def binarize_gripper(cmd):
    """정책이 낸 9차원 액션의 그리퍼 채널(7,8)을 학습 때와 같은 두 값으로 스냅."""
    if not GRIPPER_BINARIZE:
        return cmd
    cmd = np.asarray(cmd, dtype=float).copy()
    closing = float(np.mean(cmd[7:9])) < GRIPPER_CLOSE_THRESH
    cmd[7:9] = GRIPPER_CLOSED_CMD if closing else GRIPPER_OPEN
    return cmd


TARGET_POS = [TARGET_FIXED_XY[0], TARGET_FIXED_XY[1], TARGET_Z]

# 학습 영역을 고르게 덮는 4x4 격자. scene_config가 CUBE_*_RANGE에서 계산하므로
# 영역을 바꾸면 평가 위치도 따라온다 — 예전엔 손으로 맞춰야 해서 15개 중 8개가
# 학습 영역 밖이었다(학습한 적 없는 곳이라 무조건 실패).
# 학습은 연속 랜덤이므로 이 16개는 전부 "처음 보는 정확한 좌표"다.
CUBE_XY_LIST = list(EVAL_CUBE_XY_LIST)

# 고정 지점으로 학습한 모델을 평가할 때는 학습한 그 지점에서 재봐야 의미가 있다.
# FIXED_CUBE로 큐브 위치를, N_EPISODES로 반복 횟수를 지정하면 위 격자 대신
# 그 지점만 반복 평가한다.
#   FIXED_CUBE=0.425,0.125 N_EPISODES=10 ACTION_REPEAT=3 python.bat eval_act_v5_client.py
_fixed = os.environ.get("FIXED_CUBE", "").strip()
if _fixed:
    _fx, _fy = (float(v) for v in _fixed.split(","))
    CUBE_XY_LIST = [(_fx, _fy)] * int(os.environ.get("N_EPISODES", "10"))
    print(f"[client] 고정 지점 평가 모드: cube=({_fx:.3f},{_fy:.3f}) x {len(CUBE_XY_LIST)}회")


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
    position=np.array([CUBE_XY_LIST[0][0], CUBE_XY_LIST[0][1], CUBE_Z], dtype=float),
    size=CUBE_SIZE, color=np.array(CUBE_COLOR), mass=CUBE_MASS))

marker = stage.DefinePrim("/World/PlaceMarker", "Cylinder")
UsdGeom.Cylinder(marker).GetRadiusAttr().Set(0.025)
UsdGeom.Cylinder(marker).GetHeightAttr().Set(0.002)
UsdGeom.XformCommonAPI(marker).SetTranslate(Gf.Vec3d(*TARGET_POS))
UsdGeom.Gprim(marker).GetDisplayColorAttr().Set([Gf.Vec3f(0.1, 0.8, 0.1)])

ee_path = "/World/Franka/panda_hand"
wrist_xform = stage.DefinePrim(ee_path + "/WristCam", "Xform")
wrist_cam_prim = UsdGeom.Camera.Define(stage, WRIST_CAM_PRIM)
wrist_cam_prim.GetFocalLengthAttr().Set(WRIST_FOCAL)
wrist_cam_prim.GetHorizontalApertureAttr().Set(CAM_H_APERTURE)
wrist_cam_prim.GetVerticalApertureAttr().Set(CAM_V_APERTURE)
wrist_cam_prim.GetClippingRangeAttr().Set(Gf.Vec2f(CAM_CLIP_NEAR, CAM_CLIP_FAR))
UsdGeom.XformCommonAPI(wrist_xform).SetTranslate(Gf.Vec3d(*WRIST_TRANSLATE))
UsdGeom.XformCommonAPI(wrist_xform).SetRotate(Gf.Vec3f(*WRIST_ROTATE))

over_xform = stage.DefinePrim("/World/OverheadCam", "Xform")
over_cam_prim = UsdGeom.Camera.Define(stage, OVER_CAM_PRIM)
over_cam_prim.GetFocalLengthAttr().Set(OVER_FOCAL)
over_cam_prim.GetHorizontalApertureAttr().Set(CAM_H_APERTURE)
over_cam_prim.GetVerticalApertureAttr().Set(CAM_V_APERTURE)
over_cam_prim.GetClippingRangeAttr().Set(Gf.Vec2f(CAM_CLIP_NEAR, CAM_CLIP_FAR))
UsdGeom.XformCommonAPI(over_xform).SetTranslate(Gf.Vec3d(*OVER_TRANSLATE))
UsdGeom.XformCommonAPI(over_xform).SetRotate(Gf.Vec3f(*OVER_ROTATE))

# 조명도 수집과 같은 값이어야 한다. 예전엔 여기가 1000, 수집이 1500이었다 —
# DOMAIN_RANDOMIZE=False라 정책은 1500만 본 적이 있으므로 학습 분포 밖이었다.
dome_light = UsdLux.DomeLight.Define(stage, "/World/DR_DomeLight")
dome_light.GetIntensityAttr().Set(LIGHT_INTENSITY)

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
    world.step(render=RENDER)

n_dof = franka.num_dof


# 검은/단색 프레임을 세어 둔다. 이게 0이 아니면 성공률 숫자는 믿을 수 없다.
_frame_stats = {"blank": 0, "total": 0}


def grab_rgb(cam):
    _frame_stats["total"] += 1
    try:
        rgba = cam.get_rgba()
        if rgba is None or getattr(rgba, "ndim", 0) != 3 or rgba.shape[0] < 2:
            _frame_stats["blank"] += 1
            return np.zeros((IMG_H, IMG_W, 3), dtype=np.uint8)
        img = rgba[:, :, :3]
        if img.dtype != np.uint8:
            img = (np.clip(img * 255.0, 0, 255) if float(img.max()) <= 1.0
                   else np.clip(img, 0, 255)).astype(np.uint8)
        if float(img.std()) < 1.0:      # 사실상 단색 = 렌더가 안 붙은 것
            _frame_stats["blank"] += 1
        return np.ascontiguousarray(img, dtype=np.uint8)
    except Exception:
        _frame_stats["blank"] += 1
        return np.zeros((IMG_H, IMG_W, 3), dtype=np.uint8)


def get_ee_position(franka):
    """수집 스크립트(pick_place_collect_aloha.py)의 동명 함수와 같다."""
    try:
        ee_pos, _ = franka.end_effector.get_world_pose()
        return np.array(ee_pos, dtype=np.float32)
    except Exception:
        hand = stage.GetPrimAtPath("/World/Franka/panda_hand")
        xf = UsdGeom.Xformable(hand).ComputeLocalToWorldTransform(0)
        t = xf.ExtractTranslation()
        return np.array([t[0], t[1], t[2]], dtype=np.float32)


def move_to_start():
    start_full = np.array(START_POSE, dtype=float)
    if len(start_full) < n_dof:
        cur = np.array(franka.get_joint_positions(), dtype=float)
        cur[:len(start_full)] = start_full
        start_full = cur
    for _ in range(60):
        franka.apply_action(ArticulationAction(joint_positions=start_full))
        world.step(render=RENDER)


target_pos = np.array(TARGET_POS, dtype=np.float32)
results = []

print(f"[client] starting {len(CUBE_XY_LIST)} episodes "
      f"| ACTION_REPEAT={ACTION_REPEAT} (정책 {60/ACTION_REPEAT:.0f}Hz / 물리 60Hz)"
      f" | GRIPPER_BINARIZE={GRIPPER_BINARIZE}"
      f" (닫힘 판정 <{GRIPPER_CLOSE_THRESH} → 명령 {GRIPPER_CLOSED_CMD})")
for ep, (cx, cy) in enumerate(CUBE_XY_LIST):
    world.reset()
    cube.set_world_pose(position=np.array([cx, cy, 0.025], dtype=float))
    for _ in range(10):
        world.step(render=RENDER)
    move_to_start()

    reset_remote()   # 서버 정책 상태 초기화

    max_lift = 0.025
    success = False
    min_xy_err = None
    grasp_attempted = False
    final_step = MAX_STEPS
    # 정책이 큐브 위치를 실제로 읽고 있는지 보려면 손이 어디로 가는지를 봐야 한다.
    # 큐브 위치가 달라도 아래 ee_at_close가 거의 같은 값이면, 정책은 이미지에서
    # 큐브를 못 찾고 학습 데이터의 평균 궤적을 재생하고 있다는 뜻이다.
    min_ee_cube = None
    ee_at_close = None
    cube_rel_at_close = None

    cmd = None
    for step in range(MAX_STEPS):
        jp = np.array(franka.get_joint_positions(), dtype=np.float32)[:9]
        # ACTION_REPEAT 스텝마다 한 번만 정책에 물어보고, 나머지 스텝은 같은 액션을 유지.
        if step % ACTION_REPEAT == 0 or cmd is None:
            w = grab_rgb(wrist_cam)
            o = grab_rgb(over_cam)
            if step == 0:
                # 학습 데이터(bc_data_v9의 images_*)의 mean/std와 비교해서 조명·색이
                # 같은 분포인지 확인하는 용도. 크게 다르면 정책 입력이 학습 분포 밖이다.
                print(f"[client]    img wrist mean={w.mean():6.2f} std={w.std():5.2f} | "
                      f"over mean={o.mean():6.2f} std={o.std():5.2f}")
            cmd = binarize_gripper(act_remote(w, o, jp))
        franka.apply_action(ArticulationAction(joint_positions=cmd))
        world.step(render=RENDER)

        gt_cube, _ = cube.get_world_pose()
        gt_cube = np.array(gt_cube, dtype=np.float32)
        max_lift = max(max_lift, float(gt_cube[2]))
        cube_target_xy = float(np.linalg.norm(gt_cube[:2] - target_pos[:2]))
        min_xy_err = cube_target_xy if min_xy_err is None else min(min_xy_err, cube_target_xy)
        ee = get_ee_position(franka)
        ee_cube = float(np.linalg.norm(ee[:2] - gt_cube[:2]))
        min_ee_cube = ee_cube if min_ee_cube is None else min(min_ee_cube, ee_cube)
        if jp[7] < GRIPPER_CLOSING_RAW_THRESH:
            if ee_at_close is None:
                ee_at_close = [round(float(ee[0]), 3), round(float(ee[1]), 3)]
                # 수집 데이터의 obs[18:21](cube_rel = cube - ee)과 같은 양. z까지
                # 봐야 한다 — XY가 맞아도 손이 큐브보다 높으면 허공을 잡는다.
                cube_rel_at_close = [round(float(v), 4) for v in (gt_cube - ee)]
            grasp_attempted = True
        gripper_open = jp[7] > GRIPPER_CLOSING_RAW_THRESH

        if cube_target_xy < SUCCESS_XY_TOL and max_lift > (0.025 + SUCCESS_MIN_LIFT) and gripper_open:
            success = True
            final_step = step
            break

    results.append({
        "episode": ep, "cube_xy": [cx, cy], "success": success, "steps": final_step,
        "max_lift": round(max_lift, 4), "min_xy_err": round(float(min_xy_err), 4),
        "grasp_attempted": grasp_attempted,
        "min_ee_cube": round(float(min_ee_cube), 4) if min_ee_cube is not None else None,
        "ee_at_close": ee_at_close,
        "cube_rel_at_close": cube_rel_at_close,
    })
    print(f"[client] ep {ep:2d} cube=({cx:.3f},{cy:.3f}) success={success} steps={final_step} "
          f"max_lift={max_lift:.3f} min_xy_err={min_xy_err:.3f} grasp_attempted={grasp_attempted} "
          f"min_ee_cube={min_ee_cube:.3f} cube_rel_at_close={cube_rel_at_close}")

n_success = sum(r["success"] for r in results)
summary = {
    "model": "ACT v5 (grid-sampled, via policy server)",
    "action_repeat": ACTION_REPEAT,
    "n_episodes": len(results),
    "n_success": n_success,
    "success_rate": n_success / len(results),
    "avg_steps_success": float(np.mean([r["steps"] for r in results if r["success"]])) if n_success else None,
    "avg_min_xy_err": float(np.mean([r["min_xy_err"] for r in results])),
    "n_grasp_attempted": sum(r["grasp_attempted"] for r in results),
    "avg_min_ee_cube": float(np.mean([r["min_ee_cube"] for r in results
                                      if r["min_ee_cube"] is not None])),
    # 손을 닫은 지점의 산포. 큐브가 20x20cm에 퍼져 있으니 정책이 큐브를 보고
    # 있다면 이 표준편차도 수 cm 나와야 한다. 1cm 미만이면 큐브와 무관하게
    # 늘 같은 곳으로 가고 있다는 뜻 = 이미지에서 위치를 못 읽는다.
    "ee_at_close_std": [
        round(float(np.std([r["ee_at_close"][i] for r in results if r["ee_at_close"]])), 4)
        for i in (0, 1)
    ] if any(r["ee_at_close"] for r in results) else None,
    "blank_camera_frames": _frame_stats["blank"],
    "camera_frames_total": _frame_stats["total"],
    "episodes": results,
}
if _frame_stats["blank"]:
    print(f"[client] ⚠ 검은 프레임 {_frame_stats['blank']}/{_frame_stats['total']}개 — "
          f"정책이 이미지를 못 본 것이므로 이 성공률은 무의미하다 (RENDER 설정 확인)")
with open("eval_results_act_v5.json", "w") as f:
    json.dump(summary, f, indent=2)

print(f"[client] DONE success_rate={summary['success_rate']:.2f} ({n_success}/{len(results)}) "
      f"grasp_attempted={summary['n_grasp_attempted']}/{len(results)}")
print(f"[client] 손-큐브 최소거리 평균={summary['avg_min_ee_cube']:.3f}m | "
      f"손 닫은 지점 표준편차={summary['ee_at_close_std']} "
      f"(큐브는 20x20cm에 퍼져 있으므로 1cm 미만이면 큐브를 못 보고 있다는 뜻)")

try:
    send_msg(sock, {"cmd": "bye"})
    recv_msg(sock)
except Exception:
    pass
sock.close()
simulation_app.close()
