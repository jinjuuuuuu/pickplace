#!/usr/bin/env python3
# cam_tune.py — GUI에서 오버헤드 카메라를 직접 잡기 위한 도구 (윈도우 데스크톱)
# ---------------------------------------------------------------------------
# 실행:
#   "C:\isaacsim\python.bat" -u "C:\Users\user\Desktop\claude_jetbot\cam_tune.py"
#
# 하는 일:
#   - 수집 스크립트(pick_place_collect_aloha.py)와 똑같은 씬을 GUI로 띄운다.
#   - 큐브를 학습 영역(20x20cm)의 네 모서리 + 중앙으로 3초마다 옮긴다.
#   - 그때마다 오버헤드 카메라에 큐브가 몇 픽셀로 보이는지 터미널에 찍는다.
#   - 카메라의 현재 위치/회전/focal도 같이 찍는다 -> 값이 정해지면 그대로 복사.
#
# 쓰는 법:
#   1) GUI 왼쪽 Stage 트리에서 /World/OverheadCam 을 선택한다.
#   2) 오른쪽 Property 패널의 Translate / Rotate 를 조정한다.
#      (또는 뷰포트에서 기즈모로 끌어도 된다)
#   3) 터미널의 "빨강 N px"를 본다. 다섯 위치 전부 30px 이상이면 좋은 값이다.
#      한 곳이라도 0px면 그 위치는 학습이 안 된다.
#   4) 만족하면 터미널에 찍힌 OVER_TRANSLATE / OVER_ROTATE 값을 알려주면 된다.
#
# 주의: 이 스크립트는 팔을 시작 자세에 세워두기만 한다. "팔이 리치하는 중에
#       큐브가 가려지는지"는 여기서 안 보인다 — 그건 cam_check_current.py 로
#       확인해야 한다(정답 컨트롤러로 실제 파지 궤적을 돌린다).
# ---------------------------------------------------------------------------

# === 내가 정한 값 (해상도/focal) =============================================
# 해상도: 160x120 -> 320x240. 기존엔 5cm 큐브가 오버헤드에서 6px밖에 안 됐다.
#   (참고: Deepkar/redblock-pick-place-50ep 등 성공 사례는 480x640을 쓴다.
#    480x640은 에피소드당 npz가 2GB로 불어나 50개면 100GB라 여기선 무리다.)
IMG_W, IMG_H = 320, 240

# focal: 24 -> 48mm. 초점거리를 2배로 하면 보는 범위가 절반이 된다.
#   24mm: 테이블 1.29 x 0.97m 를 본다 -> 의미 있는 영역이 화면의 3%
#   48mm: 0.64 x 0.48m -> 20x20cm 작업영역이 화면의 13%, 큐브 픽셀은 2배
# 해상도 2배 x 화각 절반 = 큐브가 6px -> 약 26px (면적으로 16배)
OVER_FOCAL  = 48.0
WRIST_FOCAL = 16.0   # 손목캠은 그대로. 이미 큐브를 18~31px로 보고 있다.

# === GUI에서 조정할 값 (현재값 = 수집 때 쓴 값) ==============================
OVER_TRANSLATE = (0.4, 0.0, 1.5)
OVER_ROTATE    = (0.0, 0.0, -89.9)
WRIST_TRANSLATE = (0.15, 0.0, 0.0)
WRIST_ROTATE    = (-45.0, 179.9, -89.9)

LIGHT_INTENSITY = 1500.0   # 수집 스크립트와 동일해야 한다
HOLD_SECONDS    = 3.0      # 큐브 한 위치를 몇 초 보여줄지

# 학습 영역(pick_place_collect_aloha.py의 CUBE_*_RANGE)의 모서리 + 중앙
CUBE_POSITIONS = [
    ("중앙  ", 0.425, 0.150),
    ("좌하  ", 0.325, 0.050),
    ("좌상  ", 0.325, 0.250),
    ("우하  ", 0.525, 0.050),
    ("우상  ", 0.525, 0.250),
]
TARGET_POS = (0.500, -0.150, 0.025)
START_POSE = [0.0, -0.3, 0.0, -2.5, 0.0, 2.2, 0.8, 0.04, 0.04]
CUBE_Z = 0.025

WRIST_CAM_PRIM = "/World/Franka/panda_hand/WristCam/Camera"
OVER_CAM_PRIM  = "/World/OverheadCam/Camera"
RED_MARGIN = 40
# =============================================================================

import numpy as np
from isaacsim import SimulationApp
simulation_app = SimulationApp({"renderer": "RayTracedLighting", "headless": False})

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
    position=np.array([CUBE_POSITIONS[0][1], CUBE_POSITIONS[0][2], CUBE_Z]),
    size=0.05, color=np.array([0.8, 0.2, 0.1]), mass=0.1))

marker = stage.DefinePrim("/World/PlaceMarker", "Cylinder")
UsdGeom.Cylinder(marker).GetRadiusAttr().Set(0.025)
UsdGeom.Cylinder(marker).GetHeightAttr().Set(0.002)
UsdGeom.XformCommonAPI(marker).SetTranslate(Gf.Vec3d(*TARGET_POS))
UsdGeom.Gprim(marker).GetDisplayColorAttr().Set([Gf.Vec3f(0.1, 0.8, 0.1)])

wrist_xform = stage.DefinePrim("/World/Franka/panda_hand/WristCam", "Xform")
wrist_cam_prim = UsdGeom.Camera.Define(stage, WRIST_CAM_PRIM)
wrist_cam_prim.GetFocalLengthAttr().Set(WRIST_FOCAL)
wrist_cam_prim.GetHorizontalApertureAttr().Set(20.955)
wrist_cam_prim.GetVerticalApertureAttr().Set(15.716)
wrist_cam_prim.GetClippingRangeAttr().Set(Gf.Vec2f(0.05, 100.0))
UsdGeom.XformCommonAPI(wrist_xform).SetTranslate(Gf.Vec3d(*WRIST_TRANSLATE))
UsdGeom.XformCommonAPI(wrist_xform).SetRotate(Gf.Vec3f(*WRIST_ROTATE))

over_xform = stage.DefinePrim("/World/OverheadCam", "Xform")
over_cam_prim = UsdGeom.Camera.Define(stage, OVER_CAM_PRIM)
over_cam_prim.GetFocalLengthAttr().Set(OVER_FOCAL)
over_cam_prim.GetHorizontalApertureAttr().Set(20.955)
over_cam_prim.GetVerticalApertureAttr().Set(15.716)
over_cam_prim.GetClippingRangeAttr().Set(Gf.Vec2f(0.05, 100.0))
UsdGeom.XformCommonAPI(over_xform).SetTranslate(Gf.Vec3d(*OVER_TRANSLATE))
UsdGeom.XformCommonAPI(over_xform).SetRotate(Gf.Vec3f(*OVER_ROTATE))

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
    world.step(render=True)

n_dof = franka.num_dof
start_full = np.array(START_POSE, dtype=float)
if len(start_full) < n_dof:
    cur = np.array(franka.get_joint_positions(), dtype=float)
    cur[:len(start_full)] = start_full
    start_full = cur


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


def red_report(img):
    r, g, b = (img[:, :, i].astype(np.int16) for i in range(3))
    m = (r - np.maximum(g, b)) > RED_MARGIN
    n = int(m.sum())
    if not n:
        return f"빨강     0px  ← 안 보인다!"
    ys, xs = np.where(m)
    return (f"빨강 {n:5d}px  bbox {xs.max()-xs.min()+1:3d}x{ys.max()-ys.min()+1:3d}px  "
            f"중심=({xs.mean():4.0f},{ys.mean():4.0f})")


def read_cam_transform():
    t, r, _s, _p, _o = UsdGeom.XformCommonAPI(over_xform).GetXformVectors(0)
    focal = over_cam_prim.GetFocalLengthAttr().Get()
    return tuple(round(float(v), 4) for v in t), tuple(round(float(v), 4) for v in r), focal


print("\n" + "=" * 78)
print("  GUI 왼쪽 Stage 트리에서 /World/OverheadCam 을 선택하고 Translate/Rotate 조정")
print(f"  해상도 {IMG_W}x{IMG_H} | 오버헤드 focal {OVER_FOCAL}mm | 조명 {LIGHT_INTENSITY}")
print("  다섯 위치 전부 30px 이상이면 좋은 값. 하나라도 0px면 그 위치는 학습이 안 된다.")
print("  종료: 터미널에서 Ctrl+C")
print("=" * 78 + "\n")

steps_per_pos = int(HOLD_SECONDS * 60)
i = 0
try:
    while simulation_app.is_running():
        name, cx, cy = CUBE_POSITIONS[i % len(CUBE_POSITIONS)]
        cube.set_world_pose(position=np.array([cx, cy, CUBE_Z], dtype=float))

        for s in range(steps_per_pos):
            franka.apply_action(ArticulationAction(joint_positions=start_full))
            world.step(render=True)
            if s == steps_per_pos - 1:      # 자세가 안정된 뒤 측정
                t, r, focal = read_cam_transform()
                print(f"[{name}] cube=({cx:.3f},{cy:.3f})")
                print(f"          over  {red_report(grab_rgb(over_cam))}")
                print(f"          wrist {red_report(grab_rgb(wrist_cam))}")
                print(f"          OVER_TRANSLATE = {t}   OVER_ROTATE = {r}   focal={focal}")
        i += 1
except KeyboardInterrupt:
    t, r, focal = read_cam_transform()
    print("\n[cam_tune] 최종 값:")
    print(f"  OVER_TRANSLATE = {t}")
    print(f"  OVER_ROTATE    = {r}")
    print(f"  OVER_FOCAL     = {focal}")
    print(f"  IMG_W, IMG_H   = {IMG_W}, {IMG_H}")

simulation_app.close()
