# cam_check_current.py — pick_place_collect.py / act_hub_deploy_isaac.py 가 실제로 쓰는
# 카메라 설정(TRANSLATE/ROTATE/FOCAL) 그대로, 정답 컨트롤러(PickPlaceController)로
# 파지에 "성공"하는 궤적을 돌리며 손목캠/오버헤드캠이 그 순간 뭘 보는지 캡처한다.
# ACT 모델은 전혀 쓰지 않는다 — 순수하게 "카메라가 큐브/그리퍼를 제대로 보는가"만 검증.
# 실행: "C:\isaacsim\python.bat" "C:\Users\user\Desktop\claude_jetbot\cam_check_current.py"
# 결과: cam_check.png = [시작|파지|들기] x [손목캠|오버헤드캠] 6컷
# ---------------------------------------------------------------------------
CUBE_POS   = (0.45, 0.10, 0.025)
TARGET_POS = (0.50, -0.15, 0.025)
START_POSE = [0.0, -0.3, 0.0, -2.5, 0.0, 2.2, 0.8, 0.04, 0.04]
IMG_W, IMG_H = 160, 120

WRIST_CAM_PRIM = "/World/Franka/panda_hand/WristCam/Camera"
OVER_CAM_PRIM  = "/World/OverheadCam/Camera"

import numpy as np
from isaacsim import SimulationApp
simulation_app = SimulationApp({"renderer": "RayTracedLighting", "headless": False})

import omni.usd
from pxr import UsdGeom, Gf, UsdLux
from isaacsim.core.api import World
from isaacsim.core.api.objects import DynamicCuboid
from isaacsim.core.utils.types import ArticulationAction
from isaacsim.robot.manipulators.examples.franka import Franka
from isaacsim.robot.manipulators.examples.franka.controllers import PickPlaceController
from isaacsim.sensors.camera import Camera
from PIL import Image, ImageDraw

world = World(stage_units_in_meters=1.0)
world.scene.add_default_ground_plane()
stage = omni.usd.get_context().get_stage()
franka = world.scene.add(Franka(prim_path="/World/Franka", name="franka"))
cube = world.scene.add(DynamicCuboid(prim_path="/World/PickCube", name="pick_cube",
        position=np.array(CUBE_POS), size=0.05, color=np.array([0.8, 0.2, 0.1]), mass=0.1))
mk = stage.DefinePrim("/World/PlaceMarker", "Cylinder")
UsdGeom.Cylinder(mk).GetRadiusAttr().Set(0.025)
UsdGeom.Cylinder(mk).GetHeightAttr().Set(0.002)
UsdGeom.XformCommonAPI(mk).SetTranslate(Gf.Vec3d(*TARGET_POS))
UsdGeom.Gprim(mk).GetDisplayColorAttr().Set([Gf.Vec3f(0.1, 0.8, 0.1)])

# --- 카메라: pick_place_collect.py / act_hub_deploy_isaac.py 와 완전히 동일한 값 ---
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
over_cam  = Camera(prim_path=OVER_CAM_PRIM,  resolution=(IMG_W, IMG_H))

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
for _ in range(60):
    franka.apply_action(ArticulationAction(joint_positions=start_full))
    world.step(render=True)


def grab(cam):
    for _ in range(3):
        world.step(render=True)
    r = cam.get_rgba()
    if r is None or getattr(r, "ndim", 0) != 3 or r.shape[0] < 2:
        return np.zeros((IMG_H, IMG_W, 3), np.uint8)
    return r[:, :, :3].astype(np.uint8)


shots_w = {"시작": grab(wrist_cam)}
shots_o = {"시작": grab(over_cam)}

ctrl = PickPlaceController(name="c", gripper=franka.gripper, robot_articulation=franka)
place = np.array(TARGET_POS, dtype=float)
place[2] = 0.045
got_grasp = False
got_lift = False
for step in range(1200):
    cp, _ = cube.get_world_pose()
    act = ctrl.forward(picking_position=np.array(cp, dtype=float),
                        placing_position=place,
                        current_joint_positions=np.array(franka.get_joint_positions(), dtype=float))
    franka.apply_action(act)
    world.step(render=True)
    jq = np.array(franka.get_joint_positions())
    if (not got_grasp) and jq[7] < 0.035:
        shots_w["파지"] = grab(wrist_cam); shots_o["파지"] = grab(over_cam); got_grasp = True
    if got_grasp and (not got_lift) and float(cp[2]) > 0.12:
        shots_w["들기"] = grab(wrist_cam); shots_o["들기"] = grab(over_cam); got_lift = True
        break

if "파지" not in shots_w:
    shots_w["파지"] = grab(wrist_cam); shots_o["파지"] = grab(over_cam)
    print("[cam_check] 경고: 컨트롤러가 파지(그리퍼 닫힘)까지 못 갔음")
if "들기" not in shots_w:
    shots_w["들기"] = grab(wrist_cam); shots_o["들기"] = grab(over_cam)
    print("[cam_check] 경고: 컨트롤러가 들어올리기까지 못 갔음")

order = ["시작", "파지", "들기"]
W, H = IMG_W, IMG_H
canvas = Image.new("RGB", (W * 3, (H + 22) * 2), (20, 20, 20))
dr = ImageDraw.Draw(canvas)
for i, k in enumerate(order):
    canvas.paste(Image.fromarray(shots_w[k]), (i * W, 22))
    dr.text((i * W + 6, 4), f"손목캠-{k}", fill=(255, 255, 0))
    canvas.paste(Image.fromarray(shots_o[k]), (i * W, H + 44))
    dr.text((i * W + 6, H + 26), f"오버헤드-{k}", fill=(0, 255, 255))

out = r"C:\Users\user\Desktop\claude_jetbot\cam_check.png"
canvas.save(out)
print(f"[cam_check] 저장: {out}")
print(f"[cam_check] 파지컷 손목캠 밝기 mean={shots_w['파지'].mean():.0f} (0에 가까우면 까맣음=화각 밖/충돌)")
print(f"[cam_check] 그리퍼 파지판정: got_grasp={got_grasp} got_lift={got_lift}")
simulation_app.close()
