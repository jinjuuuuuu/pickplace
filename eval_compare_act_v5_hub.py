# eval_compare_act_v5_hub.py - HF 허브의 v5 ACT 정책(jamongsteak/act_pickplace_v5)을
# 새 큐브 위치에서 평가. Windows 데스크톱 Isaac Sim 단일 프로세스에서 돌아간다.
# ---------------------------------------------------------------------------
# eval_compare_act_v5.py(워크스테이션, 로컬 체크포인트 + 소켓 분리)와의 차이:
#   - 데스크톱 Isaac Sim 의 lerobot(0.4.4)이 v5 체크포인트를 그대로 읽을 수 있어서
#     정책 서버/클라이언트 분리 없이 한 프로세스로 실행한다.
#   - 사내 프록시가 huggingface.co 의 python SSL 을 막으므로, 모델은 미리
#     `git clone https://huggingface.co/jamongsteak/act_pickplace_v5` 로 받아둔
#     로컬 폴더를 MODEL_PATH(환경변수 ACT_V5_PATH)로 넘긴다. git 은 schannel 을
#     쓰므로 프록시를 통과한다.
#   - 성공 판정을 둘로 나눠 기록: pick(큐브를 실제로 들어올림) / place(타겟까지 옮김).
#     "집기 성공률"은 pick 기준, v4 와 비교할 숫자는 place 기준(v4 스크립트와 동일 정의).
#
# 실행:
#   set ACT_V5_PATH=<clone>\pretrained_model
#   C:\isaacsim\python.bat eval_compare_act_v5_hub.py [--n-extra 15] [--seed 456]
# ---------------------------------------------------------------------------
import argparse
import json
import os
import random
import time

import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--n-extra", type=int, default=0,
                help="v4 비교용 15개 위치 외에 추가로 뽑을 랜덤 위치 개수")
ap.add_argument("--seed", type=int, default=456, help="추가 랜덤 위치 시드")
ap.add_argument("--max-steps", type=int, default=1500)
ap.add_argument("--out", default="eval_results_act_v5_hub.json")
ap.add_argument("--save-frames", default=None,
                help="지정하면 ep0 의 카메라 프레임을 이 폴더에 PNG 로 저장(육안 확인용)")
ap.add_argument("--grid", action="store_true",
                help="새 위치 대신 '학습에 쓴 5x5 격자점'에서 평가(in-distribution 대조군). "
                     "여기서도 실패하면 일반화 문제가 아니라 다른 데 문제가 있다는 뜻.")
ap.add_argument("--action-repeat", type=int, default=1,
                help="학습 데이터를 N프레임마다 솎았다면 N을 준다(subsample_dataset.py --stride). "
                     "물리는 60Hz인데 정책은 60/N Hz로 판단하도록 학습됐으므로 액션 하나를 "
                     "N스텝 유지해야 로봇이 학습 때와 같은 속도로 움직인다.")
ap.add_argument("--no-gripper-binarize", dest="gripper_binarize", action="store_false",
                help="그리퍼 이진화를 끈다(기본은 켜짐). 학습 데이터의 그리퍼 값은 "
                     "0.025/0.04 두 개뿐인데 ACT는 회귀 모델이라 그 사이 값을 내놓고, "
                     "큐브가 5cm라 조금만 벌어져도 아예 못 잡는다.")
args = ap.parse_args()

ACTION_REPEAT = max(1, args.action_repeat)
GRIPPER_CLOSED, GRIPPER_OPEN = 0.025, 0.04
GRIPPER_MID = (GRIPPER_CLOSED + GRIPPER_OPEN) / 2.0


def binarize_gripper(cmd):
    """정책이 낸 9차원 액션의 그리퍼 채널(7,8)을 학습 때와 같은 두 값으로 스냅."""
    if not args.gripper_binarize:
        return cmd
    cmd = np.asarray(cmd, dtype=float).copy()
    cmd[7:9] = GRIPPER_CLOSED if float(np.mean(cmd[7:9])) < GRIPPER_MID else GRIPPER_OPEN
    return cmd

MODEL_PATH = os.environ.get("ACT_V5_PATH", "").strip()
TASK = "pick up the cube and place it on the target"
HEADLESS = True
# ⚠ headless 여도 render=True 여야 카메라 프레임이 갱신된다. 수집 스크립트
# (pick_place_collect.py)는 RENDER=True 로 돌았는데, 기존 평가 스크립트들은
# `render=not HEADLESS` → False 로 돌려서 정책에 정지/검은 프레임을 먹이고
# 있었다(= v4/BC 의 0% 결과는 이 하네스 버그의 산물).
RENDER = True
MAX_STEPS = args.max_steps
IMG_W, IMG_H = 160, 120

TARGET_POS = [0.50, -0.15, 0.025]
START_POSE = [0.0, -0.3, 0.0, -2.5, 0.0, 2.2, 0.8, 0.04, 0.04]

CUBE_Z = 0.025          # 큐브 반높이 = 바닥에 놓였을 때의 중심 z
PICK_MIN_LIFT = 0.02    # 중심 z 가 2cm 이상 올라가면 '집어서 들었다'로 본다
SUCCESS_XY_TOL = 0.05
SUCCESS_MIN_LIFT = 0.04

# v4/v5 평가와 동일한 15개 위치 (seed=123, 타겟서 >=0.15m) → 직접 비교용.
CUBE_XY_LIST = [
    (0.344, 0.156), (0.505, 0.195), (0.485, 0.065), (0.500, 0.009), (0.424, 0.041),
    (0.346, -0.243), (0.418, 0.114), (0.530, 0.063), (0.529, 0.182), (0.355, 0.183),
    (0.499, 0.183), (0.375, 0.014), (0.318, 0.042), (0.359, 0.132), (0.343, -0.094),
]
N_BASE = len(CUBE_XY_LIST)

# in-distribution 대조군: pick_place_collect.py 의 학습 격자점과 동일
# (CUBE_X_RANGE=(0.30,0.55) x CUBE_Y_RANGE=(-0.25,0.25), 5x5, 지점당 12 에피소드 학습).
if args.grid:
    gxs = np.linspace(0.30, 0.55, 5)
    gys = np.linspace(-0.25, 0.25, 5)
    CUBE_XY_LIST = [(round(float(x), 4), round(float(y), 4)) for x in gxs for y in gys]
    N_BASE = len(CUBE_XY_LIST)

# 추가 랜덤 위치: 수집 워크스페이스(pick_place_collect.py 와 동일 범위)에서 뽑고,
# 타겟과 0.15m 이상 떨어진 것만 채택.
if args.n_extra > 0:
    rng = random.Random(args.seed)
    extra = []
    while len(extra) < args.n_extra:
        x = rng.uniform(0.30, 0.55)
        y = rng.uniform(-0.25, 0.25)
        if np.hypot(x - TARGET_POS[0], y - TARGET_POS[1]) < 0.15:
            continue
        extra.append((round(x, 3), round(y, 3)))
    CUBE_XY_LIST = CUBE_XY_LIST + extra

WRIST_CAM_PRIM = "/World/Franka/panda_hand/WristCam/Camera"
OVER_CAM_PRIM = "/World/OverheadCam/Camera"

if not MODEL_PATH or not os.path.isdir(MODEL_PATH):
    raise SystemExit(
        "[eval_act_v5] 모델 폴더를 찾을 수 없습니다. 환경변수 ACT_V5_PATH 에\n"
        "  git clone https://huggingface.co/jamongsteak/act_pickplace_v5\n"
        "로 받은 폴더의 pretrained_model 경로를 넣어주세요. (현재값: %r)" % MODEL_PATH)

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

print(f"[eval_act_v5] loading policy from {MODEL_PATH}")
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
    position=np.array([CUBE_XY_LIST[0][0], CUBE_XY_LIST[0][1], CUBE_Z], dtype=float), size=0.05,
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
    world.step(render=RENDER)

n_dof = franka.num_dof


_blank_frames = 0


def grab_rgb(cam):
    global _blank_frames
    try:
        rgba = cam.get_rgba()
        if rgba is None or getattr(rgba, "ndim", 0) != 3 or rgba.shape[0] < 2:
            # render=False 로 스텝하면 여기로 빠진다(빈 배열) → 정책이 검은 이미지를 받는다.
            # 실제로 발생하면 결과를 신뢰할 수 없으니 세어서 요약에 남긴다.
            _blank_frames += 1
            return np.zeros((IMG_H, IMG_W, 3), dtype=np.uint8)
        img = rgba[:, :, :3]
        # 수집 스크립트와 동일한 하얀 프레임(노출 미수렴) 재시도.
        if img.dtype == np.uint8 and float(np.mean(img)) > 250:
            simulation_app.update()
            rgba2 = cam.get_rgba()
            if rgba2 is not None and getattr(rgba2, "ndim", 0) == 3 and rgba2.shape[0] >= 2:
                img = rgba2[:, :, :3]
        if img.dtype != np.uint8:
            img = (np.clip(img * 255.0, 0, 255) if float(img.max()) <= 1.0
                   else np.clip(img, 0, 255)).astype(np.uint8)
        return np.ascontiguousarray(img, dtype=np.uint8)
    except Exception:
        return np.zeros((IMG_H, IMG_W, 3), dtype=np.uint8)


def build_observation():
    # 데이터셋 state 는 joint_pos(9)만 (convert_to_lerobot.py PROPRIO_DIM=9).
    jp = np.array(franka.get_joint_positions(), dtype=np.float32)[:9]
    obs = {
        "observation.images.wrist": grab_rgb(wrist_cam),
        "observation.images.over": grab_rgb(over_cam),
        "observation.state": jp.astype(np.float32),
    }
    return obs, jp


def ee_position():
    # 그리퍼가 큐브 근처까지 갔는지 보는 진단용. 실패해도 평가는 계속.
    try:
        pos, _ = franka.end_effector.get_world_pose()
        return np.array(pos, dtype=np.float32)
    except Exception:
        return None


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
t0 = time.time()

print(f"[eval_act_v5] starting {len(CUBE_XY_LIST)} episodes "
      f"({N_BASE} v4-비교 위치 + {len(CUBE_XY_LIST) - N_BASE} 신규 랜덤)")
for ep, (cx, cy) in enumerate(CUBE_XY_LIST):
    world.reset()
    cube.set_world_pose(position=np.array([cx, cy, CUBE_Z], dtype=float))
    for _ in range(10):
        world.step(render=RENDER)
    move_to_start()

    policy.reset()
    preprocessor.reset()
    postprocessor.reset()

    cube_xy0 = np.array([cx, cy], dtype=np.float32)
    max_lift = CUBE_Z
    max_cube_shift = 0.0
    min_ee_cube = None
    min_xy_err = None
    place_success = False
    grasp_attempted = False
    final_step = MAX_STEPS
    ep_t0 = time.time()

    cmd = None
    for step in range(MAX_STEPS):
        obs, jp = build_observation()
        if args.save_frames and ep == 0 and step in (0, 200):
            # 정책이 실제로 보는 화면을 눈으로 확인할 수 있게 한 장씩 저장.
            try:
                from PIL import Image
                for tag in ("wrist", "over"):
                    Image.fromarray(obs[f"observation.images.{tag}"]).save(
                        os.path.join(args.save_frames, f"ep0_step{step}_{tag}.png"))
            except Exception as e:
                print(f"[eval_act_v5] frame save skipped: {e}", flush=True)
        # ACTION_REPEAT 스텝마다 한 번만 정책에 묻고, 사이 스텝은 같은 액션을 유지.
        if step % ACTION_REPEAT == 0 or cmd is None:
            action = predict_action(
                observation=obs, policy=policy, device=device,
                preprocessor=preprocessor, postprocessor=postprocessor,
                use_amp=policy.config.use_amp, task=TASK, robot_type="franka",
            )
            cmd = binarize_gripper(action.squeeze(0).to("cpu").numpy().astype(float))
        franka.apply_action(ArticulationAction(joint_positions=cmd))
        world.step(render=RENDER)

        gt_cube, _ = cube.get_world_pose()
        gt_cube = np.array(gt_cube, dtype=np.float32)
        max_lift = max(max_lift, float(gt_cube[2]))
        max_cube_shift = max(max_cube_shift, float(np.linalg.norm(gt_cube[:2] - cube_xy0)))
        cube_target_xy = float(np.linalg.norm(gt_cube[:2] - target_pos[:2]))
        min_xy_err = cube_target_xy if min_xy_err is None else min(min_xy_err, cube_target_xy)

        ee = ee_position()
        if ee is not None:
            d = float(np.linalg.norm(ee - gt_cube))
            min_ee_cube = d if min_ee_cube is None else min(min_ee_cube, d)

        if jp[7] < 0.035:
            grasp_attempted = True
        gripper_open = jp[7] > 0.035

        # place 성공 = v4 스크립트와 동일 정의(타겟 위 5cm 내 + 들었던 적 있음 + 그리퍼 열림)
        if cube_target_xy < SUCCESS_XY_TOL and max_lift > (CUBE_Z + SUCCESS_MIN_LIFT) and gripper_open:
            place_success = True
            final_step = step
            break

    pick_success = max_lift > (CUBE_Z + PICK_MIN_LIFT)
    results.append({
        "episode": ep,
        "cube_xy": [cx, cy],
        "novel_extra": ep >= N_BASE,
        "pick_success": bool(pick_success),
        "place_success": bool(place_success),
        "steps": final_step,
        "max_lift": round(max_lift, 4),
        "max_cube_shift": round(max_cube_shift, 4),
        "min_xy_err": round(float(min_xy_err), 4),
        "min_ee_cube_dist": None if min_ee_cube is None else round(min_ee_cube, 4),
        "grasp_attempted": grasp_attempted,
        "wall_s": round(time.time() - ep_t0, 1),
    })
    r = results[-1]
    print(f"[eval_act_v5] ep {ep:2d} cube=({cx:.3f},{cy:.3f}) pick={pick_success} place={place_success} "
          f"steps={final_step} max_lift={max_lift:.3f} shift={max_cube_shift:.3f} "
          f"min_ee_cube={r['min_ee_cube_dist']} min_xy_err={min_xy_err:.3f} "
          f"grasp={grasp_attempted} {r['wall_s']}s", flush=True)

    # 중간 결과도 계속 남겨서, 중단돼도 여기까지의 성적은 읽을 수 있게.
    with open(args.out, "w") as f:
        json.dump({"partial": True, "n_done": len(results), "episodes": results}, f, indent=2)


def rate(rows, key):
    return (sum(r[key] for r in rows) / len(rows)) if rows else None


base = [r for r in results if not r["novel_extra"]]
extra = [r for r in results if r["novel_extra"]]
n_pick = sum(r["pick_success"] for r in results)
n_place = sum(r["place_success"] for r in results)
summary = {
    "model": "ACT v5 (hub jamongsteak/act_pickplace_v5, grid-sampled data)",
    "model_path": MODEL_PATH,
    "positions": ("training 5x5 grid (in-distribution)" if args.grid
                  else "novel: 15 v4-comparison + %d fresh random" % (len(CUBE_XY_LIST) - N_BASE)),
    "n_episodes": len(results),
    "pick_success_rate": n_pick / len(results),
    "place_success_rate": n_place / len(results),
    "n_pick_success": n_pick,
    "n_place_success": n_place,
    "pick_rate_v4_positions": rate(base, "pick_success"),
    "place_rate_v4_positions": rate(base, "place_success"),
    "pick_rate_extra_positions": rate(extra, "pick_success"),
    "place_rate_extra_positions": rate(extra, "place_success"),
    "avg_steps_place_success": (float(np.mean([r["steps"] for r in results if r["place_success"]]))
                                if n_place else None),
    "avg_min_xy_err": float(np.mean([r["min_xy_err"] for r in results])),
    "avg_min_ee_cube_dist": (float(np.mean([r["min_ee_cube_dist"] for r in results
                                            if r["min_ee_cube_dist"] is not None]))
                             if any(r["min_ee_cube_dist"] is not None for r in results) else None),
    "n_grasp_attempted": sum(r["grasp_attempted"] for r in results),
    "n_cube_touched": sum(r["max_cube_shift"] > 0.01 for r in results),
    # 0 이 아니면 카메라 프레임이 비어 정책이 검은 이미지를 받은 것 → 결과 무효.
    "blank_camera_frames": _blank_frames,
    "total_wall_s": round(time.time() - t0, 1),
    "episodes": results,
}
with open(args.out, "w") as f:
    json.dump(summary, f, indent=2)

print(f"[eval_act_v5] DONE pick={n_pick}/{len(results)} ({summary['pick_success_rate']:.1%}) "
      f"place={n_place}/{len(results)} ({summary['place_success_rate']:.1%}) "
      f"touched={summary['n_cube_touched']}/{len(results)} "
      f"grasp_attempted={summary['n_grasp_attempted']}/{len(results)} → {args.out}")
simulation_app.close()
