# cam_over_sweep.py — 오버헤드 카메라 회전값 후보를 여러개 스냅샷으로 비교
# 실행: "C:\isaacsim\python.bat" "C:\Users\user\Desktop\claude_jetbot\cam_over_sweep.py"
# 결과: cam_over_sweep.png = 후보 회전값별 스냅샷 그리드
CUBE_POS = (0.45, 0.10, 0.025)
CANDIDATES = [
    (0.0, 0.0, -89.9),     # 현재 값
    (90.0, 0.0, 0.0),
    (-90.0, 0.0, 0.0),
    (90.0, 0.0, -89.9),
    (-90.0, 0.0, -89.9),
    (180.0, 0.0, -89.9),
]
IMG_W, IMG_H = 160, 120

import numpy as np
from isaacsim import SimulationApp
simulation_app = SimulationApp({"renderer": "RayTracedLighting", "headless": False})

import omni.usd
from pxr import UsdGeom, Gf, UsdLux
from isaacsim.core.api import World
from isaacsim.core.api.objects import DynamicCuboid
from isaacsim.robot.manipulators.examples.franka import Franka
from isaacsim.sensors.camera import Camera
from PIL import Image, ImageDraw
import omni.replicator.core as rep

world = World(stage_units_in_meters=1.0)
world.scene.add_default_ground_plane()
stage = omni.usd.get_context().get_stage()
franka = world.scene.add(Franka(prim_path="/World/Franka", name="franka"))
cube = world.scene.add(DynamicCuboid(prim_path="/World/PickCube", name="pick_cube",
        position=np.array(CUBE_POS), size=0.05, color=np.array([0.8, 0.2, 0.1]), mass=0.1))
dome_light = UsdLux.DomeLight.Define(stage, "/World/DR_DomeLight")
dome_light.GetIntensityAttr().Set(1000.0)

shots = []
for i, rot in enumerate(CANDIDATES):
    cam_path = f"/World/OverTest{i}/Camera"
    xf = stage.DefinePrim(f"/World/OverTest{i}", "Xform")
    cam_prim = UsdGeom.Camera.Define(stage, cam_path)
    cam_prim.GetFocalLengthAttr().Set(24.0)
    cam_prim.GetHorizontalApertureAttr().Set(20.955)
    cam_prim.GetVerticalApertureAttr().Set(15.716)
    cam_prim.GetClippingRangeAttr().Set(Gf.Vec2f(0.05, 100.0))
    UsdGeom.XformCommonAPI(xf).SetTranslate(Gf.Vec3d(0.4, 0.0, 1.5))
    UsdGeom.XformCommonAPI(xf).SetRotate(Gf.Vec3f(*rot))
    rep.create.render_product(cam_path, resolution=(IMG_W, IMG_H))

for _ in range(50):
    simulation_app.update()

cams = []
for i in range(len(CANDIDATES)):
    c = Camera(prim_path=f"/World/OverTest{i}/Camera", resolution=(IMG_W, IMG_H))
    cams.append(c)

world.reset()
for c in cams:
    c.initialize()
for _ in range(30):
    world.step(render=True)


def grab(cam):
    r = cam.get_rgba()
    if r is None or getattr(r, "ndim", 0) != 3 or r.shape[0] < 2:
        return np.zeros((IMG_H, IMG_W, 3), np.uint8)
    return r[:, :, :3].astype(np.uint8)


for c in cams:
    shots.append(grab(c))

W, H = IMG_W, IMG_H
cols = 3
rows = (len(CANDIDATES) + cols - 1) // cols
canvas = Image.new("RGB", (W * cols, (H + 20) * rows), (20, 20, 20))
dr = ImageDraw.Draw(canvas)
for i, (rot, img) in enumerate(zip(CANDIDATES, shots)):
    r, c = divmod(i, cols)
    canvas.paste(Image.fromarray(img), (c * W, r * (H + 20) + 20))
    dr.text((c * W + 4, r * (H + 20) + 4), f"rot={rot}", fill=(255, 255, 0))

out = r"C:\Users\user\Desktop\claude_jetbot\cam_over_sweep.png"
canvas.save(out)
simulation_app.close()
