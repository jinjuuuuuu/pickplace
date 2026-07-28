# wrist_cam_preview.py — 손목캠을 '파지 순간까지' 실제로 돌려보며 위치/각도 맞추기
# 실행: "C:\isaacsim\python.bat" "C:\Users\user\Desktop\claude_jetbot\wrist_cam_preview.py"
# 결과: wrist_view.png = [시작 | 파지 | 들기] 3컷. 파지 컷이 안 까맣고 양쪽 손가락이
#        보이면 성공. 아래 WRIST_* 세 줄만 바꿔가며 재실행해서 맞춘다.
# ---------------------------------------------------------------------------

# ===== 여기 세 줄만 조정 =============================================
# 손 프레임(panda_hand) 기준. +Z=손가락(접근)방향. 회전 XYZ 오일러(도).
#  - 손가락 끝에서 '뒤로 빼면'(Z를 줄이면) 물체와 거리 확보 → 파지 때 덜 까매짐.
#  - FOCAL 작을수록 화각 넓음 → 양쪽 손가락 다 들어옴.
WRIST_TRANSLATE = (0.0, 0.0, 0.0)      # (x,y,z) m — 손 원점(끝보다 뒤 = standoff↑). 손몸통 가리면 x나 y로 살짝 밀기
WRIST_ROTATE    = (180.0, 0.0, 0.0)    # (rx,ry,rz) deg — 180=정면(손가락 방향)을 봄
WRIST_FOCAL     = 10.0                  # mm — 작을수록 넓게 (양쪽 손가락 프레이밍)
# =====================================================================

CUBE_POS   = (0.45, 0.0, 0.025)
TARGET_POS = (0.50, -0.15, 0.025)
START_POSE = [0.0, -0.3, 0.0, -2.5, 0.0, 2.2, 0.8, 0.04, 0.04]
RES = (320, 240)

import numpy as np
from isaacsim import SimulationApp
simulation_app = SimulationApp({"renderer": "RayTracedLighting", "headless": False})

import omni.usd
from pxr import UsdGeom, Gf
from isaacsim.core.api import World
from isaacsim.core.api.objects import DynamicCuboid
from isaacsim.core.utils.types import ArticulationAction
from isaacsim.robot.manipulators.examples.franka import Franka
from isaacsim.robot.manipulators.examples.franka.controllers import PickPlaceController
from isaacsim.sensors.camera import Camera
from PIL import Image

world = World(stage_units_in_meters=1.0)
world.scene.add_default_ground_plane()
stage = omni.usd.get_context().get_stage()
franka = world.scene.add(Franka(prim_path="/World/Franka", name="franka"))
cube = world.scene.add(DynamicCuboid(prim_path="/World/PickCube", name="pick_cube",
        position=np.array(CUBE_POS), size=0.05, color=np.array([0.8,0.2,0.1]), mass=0.1))
mk = stage.DefinePrim("/World/PlaceMarker","Cylinder")
UsdGeom.Cylinder(mk).GetRadiusAttr().Set(0.025); UsdGeom.Cylinder(mk).GetHeightAttr().Set(0.002)
UsdGeom.XformCommonAPI(mk).SetTranslate(Gf.Vec3d(*TARGET_POS))
UsdGeom.Gprim(mk).GetDisplayColorAttr().Set([Gf.Vec3f(0.1,0.8,0.1)])

ee = "/World/Franka/panda_hand"
wx = stage.DefinePrim(ee+"/WristCam","Xform")
wc = UsdGeom.Camera.Define(stage, ee+"/WristCam/Camera")
wc.GetFocalLengthAttr().Set(WRIST_FOCAL); wc.GetHorizontalApertureAttr().Set(20.955)
wc.GetVerticalApertureAttr().Set(15.716); wc.GetClippingRangeAttr().Set(Gf.Vec2f(0.02,100.0))
UsdGeom.XformCommonAPI(wx).SetTranslate(Gf.Vec3d(*WRIST_TRANSLATE))
UsdGeom.XformCommonAPI(wx).SetRotate(Gf.Vec3f(*WRIST_ROTATE))
wrist_cam = Camera(prim_path=ee+"/WristCam/Camera", resolution=RES)

world.reset(); wrist_cam.initialize()
num_dof = len(franka.get_joint_positions())

def grab():
    for _ in range(3): world.step(render=True)
    r = wrist_cam.get_rgba()
    if r is None or r.ndim!=3 or r.shape[0]<2: return np.zeros((RES[1],RES[0],3),np.uint8)
    return r[:,:,:3].astype(np.uint8)

# 시작 자세
sp = np.array(START_POSE,dtype=float)
if len(sp)<num_dof:
    cur=np.array(franka.get_joint_positions(),dtype=float); cur[:len(sp)]=sp; sp=cur
for _ in range(120):
    franka.apply_action(ArticulationAction(joint_positions=sp)); world.step(render=True)
shots={"시작":grab()}

# PickPlaceController 로 실제 파지까지
ctrl = PickPlaceController(name="c", gripper=franka.gripper, robot_articulation=franka)
place = np.array(TARGET_POS,dtype=float); place[2]=0.045
got_grasp=False; got_lift=False
for step in range(1200):
    cp,_=cube.get_world_pose()
    act=ctrl.forward(picking_position=np.array(cp,dtype=float),
                     placing_position=place,
                     current_joint_positions=np.array(franka.get_joint_positions(),dtype=float))
    franka.apply_action(act); world.step(render=True)
    jq=np.array(franka.get_joint_positions())
    if (not got_grasp) and jq[7]<0.035:            # 그리퍼 닫힘 = 파지 순간
        shots["파지"]=grab(); got_grasp=True
    if got_grasp and (not got_lift) and float(cp[2])>0.12:  # 들어올림
        shots["들기"]=grab(); got_lift=True; break
if "파지" not in shots: shots["파지"]=grab()
if "들기" not in shots: shots["들기"]=grab()

# 3컷 나란히 저장
order=["시작","파지","들기"]; W,H=RES
canvas=Image.new("RGB",(W*3,H+22),(20,20,20))
from PIL import ImageDraw
dr=ImageDraw.Draw(canvas)
for i,k in enumerate(order):
    canvas.paste(Image.fromarray(shots[k]),(i*W,22))
    dr.text((i*W+6,4),k,fill=(255,255,0))
out=r"C:\Users\user\Desktop\claude_jetbot\wrist_view.png"
canvas.save(out)
print(f"[preview] 저장: {out}")
print(f"[preview] TRANSLATE={WRIST_TRANSLATE} ROTATE={WRIST_ROTATE} FOCAL={WRIST_FOCAL}")
print(f"[preview] 파지컷 밝기 mean={shots['파지'].mean():.0f} (0에 가까우면 까맣=실패)")
simulation_app.close()
