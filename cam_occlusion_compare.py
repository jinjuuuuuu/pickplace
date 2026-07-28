#!/usr/bin/env python3
# cam_occlusion_compare.py - 천장 수직뷰 vs 프론트 경사뷰 가림 비교 (PPT 캡처용)
# ---------------------------------------------------------------------------
# 실행:
#   "C:\isaacsim\python.bat" -u "C:\Users\user\Desktop\claude_jetbot\cam_occlusion_compare.py"
#   큐브 위치를 바꾸려면:  set CUBE_XY=0.525,0.050  후 실행
#
# 하는 일:
#   - 수집 스크립트와 같은 씬에 오버헤드 카메라를 '두 개' 동시에 놓는다.
#       OLD: 천장 수직뷰 (0.4,0,1.5) rot(0,0,-89.9) focal 24  160x120   <- v9까지 쓴 값
#       NEW: 프론트 경사뷰 (2,0,2)  rot(40,0,89.99) focal 60  320x240   <- 새 값
#   - 정답 컨트롤러(PickPlaceController)로 실제 파지 궤적을 한 번 돌린다.
#     ACT는 쓰지 않는다. 순수하게 "카메라가 큐브를 보는가"만 본다.
#   - 시작/접근/파지/들기 네 순간을 양쪽에서 캡처해 한 장의 PNG로 합친다.
#   - 전체 프레임 중 큐브가 보인 비율을 카메라별로 집계해 출력한다.
#
# 결과: cam_occlusion_compare.png  +  터미널의 가시성 통계
# ---------------------------------------------------------------------------
import os

HEADLESS = True
RENDER = True                     # headless에서도 True여야 카메라가 이미지를 준다

_xy = os.environ.get("CUBE_XY", "0.425,0.150")
CUBE_XY = tuple(float(v) for v in _xy.split(","))
CUBE_Z = 0.025
TARGET_POS = (0.500, -0.150, 0.025)
START_POSE = [0.0, -0.3, 0.0, -2.5, 0.0, 2.2, 0.8, 0.04, 0.04]
LIGHT_INTENSITY = 1500.0          # 수집 스크립트와 동일

OLD = {"prim": "/World/OverheadCamOld", "translate": (0.4, 0.0, 1.5),
       "rotate": (0.0, 0.0, -89.9), "focal": 24.0, "res": (160, 120),
       "label": "기존: 천장 수직뷰 160x120 focal24"}
NEW = {"prim": "/World/OverheadCamNew", "translate": (2.0, 0.0, 2.0),
       "rotate": (40.0, 0.0, 89.99), "focal": 60.0, "res": (320, 240),
       "label": "변경: 프론트 경사뷰 320x240 focal60"}

PANEL_W, PANEL_H = 480, 360        # 두 해상도를 같은 크기로 확대해 나란히 놓는다
RED_MARGIN = 40
MAX_STEPS = 1200
OUT_PNG = r"C:\Users\user\Desktop\claude_jetbot\cam_occlusion_compare.png"

import numpy as np
from isaacsim import SimulationApp
simulation_app = SimulationApp({"renderer": "RayTracedLighting", "headless": HEADLESS})

import omni.usd
from pxr import UsdGeom, Gf, UsdLux
from isaacsim.core.api import World
from isaacsim.core.api.objects import DynamicCuboid
from isaacsim.core.utils.types import ArticulationAction
from isaacsim.robot.manipulators.examples.franka import Franka
from isaacsim.robot.manipulators.examples.franka.controllers import PickPlaceController
from isaacsim.sensors.camera import Camera
from PIL import Image, ImageDraw, ImageFont

world = World(stage_units_in_meters=1.0)
world.scene.add_default_ground_plane()
stage = omni.usd.get_context().get_stage()
franka = world.scene.add(Franka(prim_path="/World/Franka", name="franka"))

cube = world.scene.add(DynamicCuboid(
    prim_path="/World/PickCube", name="pick_cube",
    position=np.array([CUBE_XY[0], CUBE_XY[1], CUBE_Z]), size=0.05,
    color=np.array([0.8, 0.2, 0.1]), mass=0.1))

mk = stage.DefinePrim("/World/PlaceMarker", "Cylinder")
UsdGeom.Cylinder(mk).GetRadiusAttr().Set(0.025)
UsdGeom.Cylinder(mk).GetHeightAttr().Set(0.002)
UsdGeom.XformCommonAPI(mk).SetTranslate(Gf.Vec3d(*TARGET_POS))
UsdGeom.Gprim(mk).GetDisplayColorAttr().Set([Gf.Vec3f(0.1, 0.8, 0.1)])


def define_overhead(cfg):
    xform = stage.DefinePrim(cfg["prim"], "Xform")
    cam_path = cfg["prim"] + "/Camera"
    cam = UsdGeom.Camera.Define(stage, cam_path)
    cam.GetFocalLengthAttr().Set(cfg["focal"])
    cam.GetHorizontalApertureAttr().Set(20.955)
    cam.GetVerticalApertureAttr().Set(15.716)
    cam.GetClippingRangeAttr().Set(Gf.Vec2f(0.05, 100.0))
    UsdGeom.XformCommonAPI(xform).SetTranslate(Gf.Vec3d(*cfg["translate"]))
    UsdGeom.XformCommonAPI(xform).SetRotate(Gf.Vec3f(*cfg["rotate"]))
    cfg["path"] = cam_path
    return cam_path


for cfg in (OLD, NEW):
    define_overhead(cfg)

dome_light = UsdLux.DomeLight.Define(stage, "/World/DR_DomeLight")
dome_light.GetIntensityAttr().Set(LIGHT_INTENSITY)

for _ in range(50):
    simulation_app.update()

import omni.replicator.core as rep
for cfg in (OLD, NEW):
    rep.create.render_product(cfg["path"], resolution=cfg["res"])
    cfg["cam"] = Camera(prim_path=cfg["path"], resolution=cfg["res"])

world.reset()
for cfg in (OLD, NEW):
    cfg["cam"].initialize()
for _ in range(15):
    world.step(render=RENDER)

n_dof = franka.num_dof
start_full = np.array(START_POSE, dtype=float)
if len(start_full) < n_dof:
    cur = np.array(franka.get_joint_positions(), dtype=float)
    cur[:len(start_full)] = start_full
    start_full = cur
for _ in range(60):
    franka.apply_action(ArticulationAction(joint_positions=start_full))
    world.step(render=RENDER)


def grab(cfg):
    w, h = cfg["res"]
    try:
        r = cfg["cam"].get_rgba()
        if r is None or getattr(r, "ndim", 0) != 3 or r.shape[0] < 2:
            return np.zeros((h, w, 3), np.uint8)
        img = r[:, :, :3]
        if img.dtype != np.uint8:
            img = (np.clip(img * 255.0, 0, 255) if float(img.max()) <= 1.0
                   else np.clip(img, 0, 255)).astype(np.uint8)
        return np.ascontiguousarray(img, dtype=np.uint8)
    except Exception:
        return np.zeros((h, w, 3), np.uint8)


def red_count(img):
    r, g, b = (img[:, :, i].astype(np.int16) for i in range(3))
    return int(((r - np.maximum(g, b)) > RED_MARGIN).sum())


def ee_pos():
    try:
        p, _ = franka.end_effector.get_world_pose()
        return np.array(p, dtype=float)
    except Exception:
        return np.zeros(3)


MOMENTS = ["시작", "접근", "파지", "들기"]
shots = {cfg["prim"]: {} for cfg in (OLD, NEW)}
counts = {cfg["prim"]: [] for cfg in (OLD, NEW)}


def capture(moment):
    for cfg in (OLD, NEW):
        img = grab(cfg)
        shots[cfg["prim"]][moment] = (img, red_count(img))


capture("시작")

ctrl = PickPlaceController(name="c", gripper=franka.gripper, robot_articulation=franka)
place = np.array(TARGET_POS, dtype=float)
place[2] = 0.045
got = {"접근": False, "파지": False, "들기": False}

print(f"\n[compare] 큐브 ({CUBE_XY[0]:.3f}, {CUBE_XY[1]:.3f}) 에서 정답 궤적 실행 중...\n")
for step in range(MAX_STEPS):
    cp, _ = cube.get_world_pose()
    cp = np.array(cp, dtype=float)
    act = ctrl.forward(picking_position=cp, placing_position=place,
                       current_joint_positions=np.array(franka.get_joint_positions(), dtype=float))
    franka.apply_action(act)
    world.step(render=RENDER)

    for cfg in (OLD, NEW):
        counts[cfg["prim"]].append(red_count(grab(cfg)))

    jq = np.array(franka.get_joint_positions())
    if not got["접근"] and float(np.linalg.norm(ee_pos()[:2] - cp[:2])) < 0.10:
        capture("접근"); got["접근"] = True
    if not got["파지"] and jq[7] < 0.035:
        capture("파지"); got["파지"] = True
    if got["파지"] and not got["들기"] and float(cp[2]) > 0.12:
        capture("들기"); got["들기"] = True
        break

for m in MOMENTS:
    if m not in shots[OLD["prim"]]:
        capture(m)
        print(f"[compare] 경고: '{m}' 순간에 도달하지 못해 마지막 프레임으로 대체했습니다")

# --- 가시성 통계: PPT에 넣을 핵심 숫자 --------------------------------------
print("[compare] 전체 궤적에서 큐브가 보인 프레임 비율")
for cfg in (OLD, NEW):
    c = np.array(counts[cfg["prim"]])
    vis = int((c > 0).sum())
    print(f"    {cfg['label']}")
    print(f"      보인 프레임 {vis}/{len(c)} ({vis / max(1, len(c)) * 100:5.1f}%)  "
          f"| 픽셀 중간값 {np.median(c):5.0f}  최대 {c.max():5.0f}")

# --- PNG 합성 ---------------------------------------------------------------
try:
    font = ImageFont.truetype(r"C:\Windows\Fonts\malgun.ttf", 18)
    font_small = ImageFont.truetype(r"C:\Windows\Fonts\malgun.ttf", 15)
except Exception:
    font = font_small = ImageFont.load_default()
    print("[compare] 맑은고딕을 못 찾아 기본 폰트를 씁니다(한글이 깨질 수 있음)")

BAR = 30
canvas = Image.new("RGB", (PANEL_W * len(MOMENTS), (PANEL_H + BAR) * 2 + BAR), (18, 18, 18))
dr = ImageDraw.Draw(canvas)
dr.text((10, 6), f"큐브 ({CUBE_XY[0]:.3f}, {CUBE_XY[1]:.3f}) - 동일 궤적, 동일 순간",
        fill=(255, 255, 255), font=font)

for row, cfg in enumerate((OLD, NEW)):
    c = np.array(counts[cfg["prim"]])
    vis = int((c > 0).sum())
    y0 = BAR + row * (PANEL_H + BAR)
    dr.text((10, y0 + 5), f"{cfg['label']}  -  큐브가 보인 프레임 "
                          f"{vis / max(1, len(c)) * 100:.1f}%",
            fill=(255, 220, 100) if row == 0 else (120, 255, 180), font=font)
    for col, m in enumerate(MOMENTS):
        img, n = shots[cfg["prim"]][m]
        panel = Image.fromarray(img).resize((PANEL_W, PANEL_H), Image.NEAREST)
        canvas.paste(panel, (col * PANEL_W, y0 + BAR))
        tag = f"{m}  -  큐브 {n}px" + ("  ← 안 보임" if n == 0 else "")
        dr.rectangle([col * PANEL_W, y0 + BAR, col * PANEL_W + PANEL_W - 1, y0 + BAR + 24],
                     fill=(0, 0, 0))
        dr.text((col * PANEL_W + 8, y0 + BAR + 3), tag,
                fill=(255, 90, 90) if n == 0 else (230, 230, 230), font=font_small)

canvas.save(OUT_PNG)
print(f"\n[compare] 저장: {OUT_PNG}")
simulation_app.close()
