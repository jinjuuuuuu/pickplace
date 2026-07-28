#!/usr/bin/env python3
# cam_tune.py - GUI에서 오버헤드 카메라를 직접 잡기 위한 도구 (윈도우 데스크톱)
# ---------------------------------------------------------------------------
# 실행:
#   "C:\isaacsim\python.bat" -u "C:\Users\user\Desktop\claude_jetbot\cam_tune.py"
#
# 하는 일:
#   - 수집 스크립트(pick_place_collect_aloha.py)와 똑같은 씬을 GUI로 띄운다.
#   - 큐브를 수집 영역(scene_config.CUBE_*_RANGE)의 네 모서리 + 중앙으로 옮긴다.
#   - 그때마다 오버헤드 카메라에 큐브가 몇 픽셀로 보이는지 터미널에 찍는다.
#   - 카메라의 현재 위치/회전/focal도 같이 찍는다 -> 값이 정해지면 그대로 복사.
#
# 쓰는 법:
#   1) GUI 왼쪽 Stage 트리에서 /World/OverheadCam 을 선택한다.
#   2) 오른쪽 Property 패널의 Translate / Rotate 를 조정한다.
#      (또는 뷰포트에서 기즈모로 끌어도 된다)
#   3) 터미널의 "빨강 N px"를 본다. 다섯 위치 전부 30px 이상이면 좋은 값이다.
#      한 곳이라도 0px면 그 위치는 학습이 안 된다.
#   4) 만족하면 터미널에 찍힌 값을 scene_config.py 의 OVER_* 에 반영한다.
#
# 주의: 이 스크립트는 팔을 시작 자세에 세워두기만 한다. "팔이 리치하는 중에
#       큐브가 가려지는지"는 여기서 안 보인다 - 그건 cam_check_current.py 로
#       확인해야 한다 - cam_occlusion_compare.py 가 정답 컨트롤러로 실제 파지
#       궤적을 돌리며 두 카메라 구도를 동시에 찍는다.
# ---------------------------------------------------------------------------

# 값은 전부 scene_config.py에서 온다. 여기서 조정한 값을 그 파일에 반영하면
# 수집·평가가 자동으로 따라온다 (예전엔 파일마다 복사돼 있어서 어긋났다).
from scene_config import (
    IMG_W, IMG_H, WRIST_CAM_PRIM, OVER_CAM_PRIM,
    CAM_H_APERTURE, CAM_V_APERTURE, CAM_CLIP_NEAR, CAM_CLIP_FAR,
    WRIST_FOCAL, WRIST_TRANSLATE, WRIST_ROTATE,
    OVER_FOCAL, OVER_TRANSLATE, OVER_ROTATE,
    LIGHT_INTENSITY, CUBE_COLOR, CUBE_SIZE, CUBE_MASS, CUBE_Z,
    TARGET_FIXED_XY, TARGET_Z, START_POSE,
    CUBE_X_RANGE, CUBE_Y_RANGE,
)

HOLD_SECONDS = 3.0      # 큐브 한 위치를 몇 초 보여줄지
RED_MARGIN = 40

# 수집 영역(CUBE_*_RANGE)의 네 모서리 + 중앙. 모서리에서 큐브가 보이지 않으면
# 그 위치는 학습이 안 된다. 영역을 바꾸면 여기도 자동으로 따라온다.
_cx = (CUBE_X_RANGE[0] + CUBE_X_RANGE[1]) / 2
_cy = (CUBE_Y_RANGE[0] + CUBE_Y_RANGE[1]) / 2
CUBE_POSITIONS = [
    ("중앙  ", _cx, _cy),
    ("좌하  ", CUBE_X_RANGE[0], CUBE_Y_RANGE[0]),
    ("좌상  ", CUBE_X_RANGE[0], CUBE_Y_RANGE[1]),
    ("우하  ", CUBE_X_RANGE[1], CUBE_Y_RANGE[0]),
    ("우상  ", CUBE_X_RANGE[1], CUBE_Y_RANGE[1]),
]
TARGET_POS = (TARGET_FIXED_XY[0], TARGET_FIXED_XY[1], TARGET_Z)
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
    size=CUBE_SIZE, color=np.array(CUBE_COLOR), mass=CUBE_MASS))

marker = stage.DefinePrim("/World/PlaceMarker", "Cylinder")
UsdGeom.Cylinder(marker).GetRadiusAttr().Set(0.025)
UsdGeom.Cylinder(marker).GetHeightAttr().Set(0.002)
UsdGeom.XformCommonAPI(marker).SetTranslate(Gf.Vec3d(*TARGET_POS))
UsdGeom.Gprim(marker).GetDisplayColorAttr().Set([Gf.Vec3f(0.1, 0.8, 0.1)])

wrist_xform = stage.DefinePrim("/World/Franka/panda_hand/WristCam", "Xform")
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
