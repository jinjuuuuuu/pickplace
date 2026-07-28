# bc_deploy_cam.py  (Isaac Sim 5.1, Windows standalone)
# ---------------------------------------------------------------------------
# bc_deploy.py 와 동일하지만, 큐브의 위치를 시뮬레이터의 '정답'(cube.get_world_pose())
# 대신 "카메라(YOLO + depth) 기반 추정값"으로 바꾼 버전.
#
# [구조]
#   이 스크립트(Isaac, Windows)
#     ├─ 장면에 top-down 카메라를 추가하고
#     ├─ ROS2 bridge 로 /rgb, /depth, /camera_info 를 발행한다
#     └─ cube_perception_node(WSL2) 가 추정한 큐브 월드좌표를 받아서 obs 에 넣는다
#
#   cube_perception_node.py(WSL2)
#     └─ /rgb,/depth,/camera_info 구독 → YOLO+depth 로 큐브 3D 추정 → /cube_pose 발행
#                                                                  → (옵션) UDP 로 이 스크립트에 전송
#
# [큐브 좌표를 받는 두 가지 방법] RECEIVE_MODE 로 선택
#   "udp"  : 가장 안정적(추가 설치 불필요). 이 스크립트가 UDP 서버로 대기,
#            perception 노드를 udp_enable:=true 로 실행하면 (x,y,z)를 직접 보내준다.
#   "ros2" : rclpy 가 이 파이썬 환경에서 import 되는 경우. /cube_pose 를 직접 구독.
#
# Run:
#   "C:\isaacsim\python.bat" "C:\Users\user\Desktop\claude_jetbot\bc_deploy_cam.py"
# ---------------------------------------------------------------------------

# === 설정 ===================================================================
POLICY_PATH    = r"C:\Users\user\Desktop\claude_jetbot\bc_data_v3\bc_policy.pt"
NORM_STATS_PATH= r"C:\Users\user\Desktop\claude_jetbot\bc_data_v3\bc_policy_norm_stats.pt"

HEADLESS       = False
ISAACSIM_PATH  = r"C:\isaacsim"
FASTDDS_XML    = r"C:\.ros\fastdds.xml"
MAX_STEPS      = 2500
REPLAN_EVERY   = 60

CUBE_POS    = [0.40, 0.15, 0.025]
TARGET_POS  = [0.50, -0.15, 0.025]

# ----- 시각 랜덤화 (수집 때처럼 큐브 색/조명 랜덤) --------------------------
# 주의: 정책은 상태(숫자)만 보므로 색/조명은 '파지'엔 영향 없음. '카메라 인식'에만 영향.
#  · 큐브 색을 랜덤화하면 color 검출기(빨강)는 못 잡음 → detector:=world +
#    프롬프트 ["cube","box","block"] 로 실행해야 색 무관하게 잡힘.
RANDOMIZE_CUBE_COLOR = True     # True 면 큐브 색을 매 실행 랜덤 (world 검출기와 함께 쓰세요)
RANDOMIZE_LIGHT      = True     # True 면 조명 밝기 랜덤

# ----- (테스트) 파지 전 큐브를 가끔 옮기기 ----------------------------------
# 카메라가 큐브를 '실시간으로' 잘 따라가는지 확인용. 카메라가 잘 따라가면 팔이
# 새 위치로 다시 조준하고, 못 따라가면 처음 위치만 보고 감.
MOVE_CUBE_BEFORE_GRASP = False
MOVE_EVERY_STEPS = 120          # 이 스텝마다 한 번 옮김(클수록 덜 자주). 너무 자주면 불안정.
MOVE_STOP_DIST   = 0.10         # 손이 큐브에 이 거리(m) 이내로 오면(=곧 집음) 그만 옮김
MOVE_X = (0.36, 0.49)           # 옮길 x 범위 (잘 집는 픽업 영역)
MOVE_Y = (0.10, 0.22)           # 옮길 y 범위 (위쪽=팔에 안 가리는 영역. 중앙/낮은 y는 팔이 가림)
MAX_MOVES = 3                   # 추적 확인 단계에서 큐브를 옮겨보는 횟수
# 마지막(3/3) 이동은 랜덤이 아니라 '확실히 집히는 고정 위치'로 둔다 → 추적은 보여주되 파지는 안정적.
FIX_LAST_MOVE = True
GRASP_POS = [0.40, 0.15, 0.025] # 파지가 잘 되는 sweet spot (최종 큐브 위치)
SETTLE_MAX_STEPS = 900          # 큐브를 옮긴 뒤 카메라 추정이 따라올 때까지 기다리는 최대 스텝
                                # (추정이 일치하면 즉시 통과 · color 검출기면 몇 프레임이면 됨)
MOVE_DWELL_STEPS = 120          # 각 이동 후 이만큼 멈춰서 큐브를 눈으로 볼 수 있게 함(GUI 확인용)

# ----- 카메라 (top-down) ----------------------------------------------------
CAMERA_PRIM   = "/World/CubeCam"
CAM_POS       = [0.45, 0.0, 1.2]    # 작업공간 위 1.2m, 회전 없음(수직 하방을 봄)
CAM_RES       = (320, 240)          # (width, height) — 낮출수록 렌더+DDS 부하↓ (WSL2 브리지 병목 완화)
# 발행 스킵: N 프레임 건너뛰고 발행 → (N+1)프레임마다 1번. 클수록 시뮬 가벼움/발행 뜸.
# 큐브는 느리게 움직이니 몇 Hz면 충분. 시뮬이 느리면 값을 키우세요.
CAM_PUBLISH_SKIP = 3                # 3 → 4프레임마다 1번 발행
CAM_FOCAL     = 24.0                # mm (USD focalLength)
CAM_H_APER    = 20.955              # mm (horizontalAperture)
CAM_V_APER    = 15.716              # mm (verticalAperture = h_aper*H/W → 정사각 픽셀, fx==fy)

# ----- 큐브 좌표 수신 --------------------------------------------------------
USE_CAMERA_CUBE = True              # False 면 기존처럼 정답좌표 사용(비교/디버그용)
RECEIVE_MODE    = "udp"             # "udp" | "ros2"
UDP_PORT        = 5599
CUBE_STALE_SEC  = 3.0               # 이 시간보다 오래된 추정값은 무시.
                                    # /rgb 가 뚝뚝 끊기면(느린 sim) 값을 키운다.
# 카메라는 큐브 '윗면'(≈0.05m)을 본다. 학습은 큐브 '중심'(0.025m)을 썼으므로
# 절반 높이만큼 내려 학습 분포와 맞춘다(안 맞추면 그리퍼가 큐브 위 허공을 잡음).
CAMERA_Z_OFFSET = -0.025
# 신선한 추정값이 없을 때: True 면 마지막 카메라값 유지(좌표 점프 방지),
# False 면 정답좌표로 폴백. 큐브는 파지 전 정지 상태라 유지가 더 안전.
HOLD_LAST_WHEN_STALE = True
# 파지 후: 큐브는 그리퍼에 물려 함께 움직인다(카메라론 안 보임). 학습 관측처럼
# "파지 순간의 큐브-손 상대위치"를 유지하며 손(ee)을 따라가게 한다.
POST_GRASP_TRACK_EE   = True
GRIPPER_CLOSED_THRESH = 0.035   # 손가락 관절(7번) < 이 값이면 닫힘으로 간주
GRASP_LATCH_DIST      = 0.08    # 손이 큐브에 이 거리(m) 이내에서 닫히면 '파지'로 판정
SUCCESS_XY_TOL        = 0.08    # 큐브가 목표 XY 이 반경(m) 안 + 그리퍼 열림이면 성공으로 종료
# 깔끔한 놓기: 큐브를 든 채 목표 XY 근처에 왔고 큐브가 이 높이까지 내려오면, 바닥에 눌러
# 열지 말고 '살짝 위'에서 그리퍼를 열어 큐브를 안 밀치게 한다.
CLEAN_RELEASE      = True
PLACE_XY_TOL       = 0.05       # 목표 XY 이 반경(m) 안일 때만 놓기(가까이 왔을 때만 → 목표서 멀리 안 떨어짐)
PLACE_RELEASE_Z    = 0.035      # 큐브(중심) 높이가 이 값 이하로 내려오면 열기(≈바닥 위 1cm → 거의 안 떨어짐)
# =============================================================================

import os, sys, time, socket, struct, threading
import numpy as np
import torch
import torch.nn as nn

if sys.platform == "win32":
    os.environ.setdefault("ROS_DISTRO",        "humble")
    os.environ.setdefault("RMW_IMPLEMENTATION","rmw_fastrtps_cpp")
    os.environ.setdefault("ROS_DOMAIN_ID",     "0")
    if os.path.exists(FASTDDS_XML):
        os.environ.setdefault("FASTRTPS_DEFAULT_PROFILES_FILE", FASTDDS_XML)
    bridge_lib = os.path.join(ISAACSIM_PATH, "exts",
                              "isaacsim.ros2.bridge", "humble", "lib")
    if os.path.isdir(bridge_lib):
        os.environ["PATH"] = bridge_lib + os.pathsep + os.environ.get("PATH", "")
        try:
            os.add_dll_directory(bridge_lib)
        except Exception:
            pass

from isaacsim import SimulationApp
simulation_app = SimulationApp({"renderer": "RayTracedLighting", "headless": HEADLESS})

import omni.usd
import omni.graph.core as og
from pxr import UsdGeom, Gf

from isaacsim.core.api import World
from isaacsim.core.api.objects import DynamicCuboid
from isaacsim.core.utils import extensions
from isaacsim.core.utils.types import ArticulationAction
from isaacsim.robot.manipulators.examples.franka import Franka

extensions.enable_extension("isaacsim.ros2.bridge")
simulation_app.update()


# ============================================================================
# 0. 큐브 좌표 수신기 (UDP 또는 ROS2)
# ============================================================================
class CubePoseReceiver:
    """perception 노드가 보낸 큐브 월드좌표(x,y,z)를 백그라운드에서 받아 캐싱."""
    def __init__(self, mode="udp", udp_port=5599):
        self.mode = mode
        self._lock = threading.Lock()
        self._pos = None
        self._t = 0.0
        self.ok = False
        if mode == "udp":
            self._start_udp(udp_port)
        elif mode == "ros2":
            self._start_ros2()

    def _set(self, xyz):
        with self._lock:
            self._pos = np.array(xyz, dtype=np.float32)
            self._t = time.time()

    def get(self, max_age):
        with self._lock:
            if self._pos is None or (time.time() - self._t) > max_age:
                return None
            return self._pos.copy()

    def get_any(self):
        with self._lock:
            return None if self._pos is None else self._pos.copy()

    # ---- UDP ----
    def _start_udp(self, port):
        def loop():
            s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            s.bind(("0.0.0.0", port))
            print(f"[recv] UDP 서버 대기: 0.0.0.0:{port}")
            self.ok = True
            while True:
                data, _ = s.recvfrom(64)
                if len(data) >= 24:
                    self._set(struct.unpack("<3d", data[:24]))
        threading.Thread(target=loop, daemon=True).start()

    # ---- ROS2 ----
    def _start_ros2(self):
        try:
            import rclpy
            from rclpy.node import Node
            from geometry_msgs.msg import PointStamped
        except Exception as e:
            print(f"[recv] !! rclpy import 실패: {e}")
            print("[recv] !! 이 파이썬 환경에서 rclpy 를 쓸 수 없습니다. "
                  "RECEIVE_MODE='udp' 로 바꾸세요.")
            return

        def loop():
            rclpy.init()
            node = rclpy.create_node("bc_deploy_cube_listener")
            node.create_subscription(
                PointStamped, "/cube_pose",
                lambda m: self._set((m.point.x, m.point.y, m.point.z)), 10)
            print("[recv] ROS2 /cube_pose 구독 시작")
            self.ok = True
            rclpy.spin(node)
        threading.Thread(target=loop, daemon=True).start()


# ============================================================================
# 1. 신경망 (bc_train.py 와 완전히 동일)
# ============================================================================
class BCChunkPolicy(nn.Module):
    def __init__(self, obs_dim, action_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, 512), nn.ReLU(),
            nn.Linear(512, 512),     nn.ReLU(),
            nn.Linear(512, 256),     nn.ReLU(),
            nn.Linear(256, action_dim),
        )
    def forward(self, obs):
        return self.net(obs)


# ============================================================================
# 2. 정책 + 정규화 로드
# ============================================================================
print("[deploy] 정책 로드 중...")
ckpt = torch.load(POLICY_PATH, map_location="cpu", weights_only=False)
OBS_DIM    = ckpt["obs_dim"]
ACTION_DIM = ckpt["action_dim"]
CHUNK_H    = ckpt["chunk_h"]
JOINT_DIM  = ckpt["joint_dim"]

policy = BCChunkPolicy(OBS_DIM, ACTION_DIM)
policy.load_state_dict(ckpt["policy"])
policy.eval()
print(f"[deploy] 정책 로드 완료 (epoch {ckpt['epoch']}, val loss {ckpt['val_loss']:.6f}, "
      f"H={CHUNK_H}, replan={REPLAN_EVERY})")

norm = torch.load(NORM_STATS_PATH, map_location="cpu", weights_only=False)
obs_mean = norm["obs_mean"].numpy()
obs_std  = norm["obs_std"].numpy()
act_mean = norm["act_mean"].numpy()
act_std  = norm["act_std"].numpy()
START_POSE = norm["start_pose"].numpy() if "start_pose" in norm else None
print("[deploy] 정규화 통계 로드 완료")


# ============================================================================
# 3. Scene 구성
# ============================================================================
world  = World(stage_units_in_meters=1.0)
world.scene.add_default_ground_plane()
stage  = omni.usd.get_context().get_stage()

franka = world.scene.add(Franka(prim_path="/World/Franka", name="franka"))

cube = world.scene.add(DynamicCuboid(
    prim_path="/World/PickCube", name="pick_cube",
    position=np.array(CUBE_POS), size=0.05,
    color=np.array([0.8, 0.2, 0.1]), mass=0.1,
))

marker = stage.DefinePrim("/World/PlaceMarker", "Cylinder")
UsdGeom.Cylinder(marker).GetRadiusAttr().Set(0.025)
UsdGeom.Cylinder(marker).GetHeightAttr().Set(0.002)
UsdGeom.XformCommonAPI(marker).SetTranslate(Gf.Vec3d(*TARGET_POS))
UsdGeom.Gprim(marker).GetDisplayColorAttr().Set([Gf.Vec3f(0.1, 0.8, 0.1)])

# ---- top-down 카메라 prim ---------------------------------------------------
# Z-up 스테이지에서 '회전 없는' 카메라는 로컬 -Z(=월드 -Z, 즉 수직 하방)를 바라본다.
# 따라서 cube_perception_node 의 외부행렬(top-down 기본값)과 정확히 일치한다.
cam = UsdGeom.Camera.Define(stage, CAMERA_PRIM)
cam.GetFocalLengthAttr().Set(CAM_FOCAL)
cam.GetHorizontalApertureAttr().Set(CAM_H_APER)
cam.GetVerticalApertureAttr().Set(CAM_V_APER)
cam.GetClippingRangeAttr().Set(Gf.Vec2f(0.05, 100.0))
cam_xform = UsdGeom.XformCommonAPI(cam)
cam_xform.SetTranslate(Gf.Vec3d(*CAM_POS))
cam_xform.SetRotate(Gf.Vec3f(0.0, 0.0, 0.0))   # 수직 하방
print(f"[deploy] 카메라 배치: {CAMERA_PRIM} @ {CAM_POS} (top-down)")


# ============================================================================
# 3b. ROS2 카메라 발행 그래프 (rgb / depth / camera_info)
# ============================================================================
def setup_ros2_camera_graph():
    keys = og.Controller.Keys
    graph_path = "/ROS_CubeCam"
    try:
        og.Controller.edit(
            {"graph_path": graph_path, "evaluator_name": "execution"},
            {
                keys.CREATE_NODES: [
                    ("OnTick",      "omni.graph.action.OnPlaybackTick"),
                    ("Context",     "isaacsim.ros2.bridge.ROS2Context"),
                    ("createRP",    "isaacsim.core.nodes.IsaacCreateRenderProduct"),
                    ("helperRgb",   "isaacsim.ros2.bridge.ROS2CameraHelper"),
                    ("helperInfo",  "isaacsim.ros2.bridge.ROS2CameraInfoHelper"),
                    ("helperDepth", "isaacsim.ros2.bridge.ROS2CameraHelper"),
                ],
                keys.CONNECT: [
                    ("OnTick.outputs:tick",              "createRP.inputs:execIn"),
                    ("createRP.outputs:execOut",         "helperRgb.inputs:execIn"),
                    ("createRP.outputs:execOut",         "helperInfo.inputs:execIn"),
                    ("createRP.outputs:execOut",         "helperDepth.inputs:execIn"),
                    ("createRP.outputs:renderProductPath","helperRgb.inputs:renderProductPath"),
                    ("createRP.outputs:renderProductPath","helperInfo.inputs:renderProductPath"),
                    ("createRP.outputs:renderProductPath","helperDepth.inputs:renderProductPath"),
                    ("Context.outputs:context",          "helperRgb.inputs:context"),
                    ("Context.outputs:context",          "helperInfo.inputs:context"),
                    ("Context.outputs:context",          "helperDepth.inputs:context"),
                ],
                keys.SET_VALUES: [
                    ("Context.inputs:useDomainIDEnvVar", True),
                    ("createRP.inputs:width",  int(CAM_RES[0])),
                    ("createRP.inputs:height", int(CAM_RES[1])),
                    ("helperRgb.inputs:frameId",   "sim_camera"),
                    ("helperRgb.inputs:topicName", "rgb"),
                    ("helperRgb.inputs:type",      "rgb"),
                    ("helperRgb.inputs:frameSkipCount",   int(CAM_PUBLISH_SKIP)),
                    ("helperInfo.inputs:frameId",   "sim_camera"),
                    ("helperInfo.inputs:topicName", "camera_info"),
                    ("helperInfo.inputs:frameSkipCount",  int(CAM_PUBLISH_SKIP)),
                    ("helperDepth.inputs:frameId",   "sim_camera"),
                    ("helperDepth.inputs:topicName", "depth"),
                    ("helperDepth.inputs:type",      "depth"),  # distance_to_image_plane (미터)
                    ("helperDepth.inputs:frameSkipCount", int(CAM_PUBLISH_SKIP)),
                ],
            },
        )
        # createRP 의 cameraPrim 타깃 연결
        from isaacsim.core.nodes.scripts.utils import set_target_prims
        set_target_prims(primPath=graph_path + "/createRP",
                         inputName="inputs:cameraPrim",
                         targetPrimPaths=[CAMERA_PRIM])
        print(f"[deploy] ROS2 카메라 그래프 생성 완료: {graph_path}")
        print("[deploy]   토픽: /rgb  /depth  /camera_info")
    except Exception as e:
        print(f"[deploy] !! ROS2 카메라 그래프 생성 실패: {e}")
        print("[deploy] !! 노드 타입 이름이 버전에 따라 다를 수 있습니다. "
              "GUI(Tools > Robotics > ROS 2 OmniGraphs > Camera)로도 만들 수 있습니다.")

setup_ros2_camera_graph()

# ---- 시각 랜덤화 (수집 때와 동일: 큐브 색 + 조명 밝기) ----------------------
import random as _random
from pxr import UsdLux
_dome_light = UsdLux.DomeLight.Define(stage, "/World/DR_DomeLight")
def randomize_domain():
    if RANDOMIZE_CUBE_COLOR:
        try:
            mat = cube.get_applied_visual_material()
            mat.set_color(np.array([_random.uniform(0, 1), _random.uniform(0, 1), _random.uniform(0, 1)]))
        except Exception as e:
            print(f"[deploy] 큐브 색 랜덤화 실패(무시): {e}")
    if RANDOMIZE_LIGHT:
        try:
            _dome_light.GetIntensityAttr().Set(_random.uniform(500.0, 2500.0))
        except Exception as e:
            print(f"[deploy] 조명 랜덤화 실패(무시): {e}")
    print(f"[deploy] 시각 랜덤화: 큐브색={'랜덤' if RANDOMIZE_CUBE_COLOR else '고정'}, "
          f"조명={'랜덤' if RANDOMIZE_LIGHT else '고정'}")

world.reset()
randomize_domain()

# ---- 워밍업: 그리퍼 열기 ----------------------------------------------------
FINGER_OPEN = 0.04
num_dof = len(franka.get_joint_positions())
finger_idx = np.array([num_dof - 2, num_dof - 1], dtype=int)
print("[deploy] 워밍업(그리퍼 열기)...")
for _ in range(200):
    franka.apply_action(ArticulationAction(
        joint_positions=np.full(2, FINGER_OPEN, dtype=float),
        joint_indices=finger_idx))
    world.step(render=not HEADLESS)
print(f"[deploy] 워밍업 완료. 손가락: "
      f"{np.round(np.array(franka.get_joint_positions())[finger_idx],4)}")

# ---- 수집과 동일한 고정 시작 자세로 이동 -----------------------------------
if START_POSE is not None:
    start_full = np.array(START_POSE, dtype=float)
else:
    start_full = np.array([0.0,-0.3,0.0,-2.5,0.0,2.2,0.8,0.04,0.04], dtype=float)
if len(start_full) < num_dof:
    cur = np.array(franka.get_joint_positions(), dtype=float)
    cur[:len(start_full)] = start_full
    start_full = cur
print(f"[deploy] 시작 자세로 이동: 팔={np.round(start_full[:7],3)}")
for _ in range(60):
    franka.apply_action(ArticulationAction(joint_positions=start_full))
    world.step(render=not HEADLESS)
print(f"[deploy] 이동 완료. 현재 팔: {np.round(np.array(franka.get_joint_positions())[:7],3)}")

target_pos = np.array(TARGET_POS, dtype=np.float32)

# ---- 큐브 좌표 수신기 시작 --------------------------------------------------
receiver = None
if USE_CAMERA_CUBE:
    print(f"[deploy] 카메라 기반 큐브 좌표 사용 (RECEIVE_MODE={RECEIVE_MODE})")
    receiver = CubePoseReceiver(mode=RECEIVE_MODE, udp_port=UDP_PORT)
    # perception 노드가 첫 추정을 보낼 때까지 '무한' 대기(시뮬은 계속 돌림).
    # 정답좌표로 때우지 않는다 — 값이 올 때까지 기다린다.
    print("[deploy] perception 노드의 첫 큐브 추정 대기 중...(정답좌표 사용 안 함)")
    waited = 0
    while receiver.get_any() is None:
        world.step(render=not HEADLESS)
        waited += 1
        if waited % 300 == 0:
            print(f"[deploy] ...대기 {waited} 스텝. perception 노드 실행/토픽 발행 확인하세요.")
    print("[deploy] 큐브 추정값 수신 확인!")
else:
    print("[deploy] 정답(get_world_pose) 큐브 좌표 사용")


# ============================================================================
# 4. 관측 구성 (수집 get_observation 과 동일한 순서/구성!)
#    단, cube_world_pos 만 카메라 추정값으로 대체
# ============================================================================
_grasp_state = {"holding": False, "offset": np.zeros(3, dtype=np.float32), "resting": None}

def _camera_cube_pos():
    """카메라 추정 큐브 위치(윗면→중심 보정). 값이 없으면 None 반환 — 정답으로 폴백하지 않는다."""
    if receiver is None:
        return None
    est = receiver.get(CUBE_STALE_SEC)
    if est is None and HOLD_LAST_WHEN_STALE:
        est = receiver.get_any()              # 파지 전 큐브는 정지 → 마지막 카메라값 유지(점프 방지)
    if est is None:
        return None
    est = est.astype(np.float32).copy()
    est[2] += CAMERA_Z_OFFSET                 # 윗면 → 중심 높이로 보정
    return est

def wait_for_camera():
    """카메라 값이 없으면 값이 올 때까지 시뮬을 돌리며 대기(정답좌표 사용 안 함)."""
    if not USE_CAMERA_CUBE or _camera_cube_pos() is not None:
        return
    print("[deploy] 카메라 추정값 끊김 → 값이 올 때까지 대기(정답좌표 사용 안 함)...")
    waited = 0
    while _camera_cube_pos() is None:
        world.step(render=not HEADLESS)
        waited += 1
        if waited % 300 == 0:
            print(f"[deploy] ...대기 {waited} 스텝. perception 노드/토픽 발행 확인.")

def compute_cube_world_pos(joint_pos, ee_pos):
    """큐브 위치 상태기계(정답좌표 안 씀):
      · 파지 전  : 카메라가 본 테이블 위 큐브
      · 파지 중  : 손(ee)+파지순간 offset 으로 손을 따라 이동
      · 놓는 순간: '놓은 자리'(손+offset)에 고정 → 이후 정지(카메라 지연 점프 방지)
    놓은 뒤 손을 따라가지도(OOD), 지연된 카메라로 튀지도 않게 해서 정답 버전과 동일하게 동작."""
    if not USE_CAMERA_CUBE:                                  # 명시적 정답 모드(비교/디버그 전용)
        gt, _ = cube.get_world_pose()
        return np.array(gt, dtype=np.float32)
    if not POST_GRASP_TRACK_EE:
        return _camera_cube_pos()

    cam = _camera_cube_pos()                                 # wait_for_camera 이후이므로 None 아님
    resting = _grasp_state["resting"]
    ref = resting if resting is not None else cam            # 현재 테이블 위 큐브의 기준 위치
    gripper_closed = joint_pos[7] < GRIPPER_CLOSED_THRESH

    if gripper_closed:
        if not _grasp_state["holding"] and np.linalg.norm(ref - ee_pos) < GRASP_LATCH_DIST:
            _grasp_state["holding"] = True
            _grasp_state["offset"]  = (ref - ee_pos).astype(np.float32)   # 파지 순간 상대위치 고정
            _grasp_state["resting"] = None                                # 집었으니 테이블엔 없음
            print(f"[deploy] 파지 감지 → 큐브가 손을 따라감 "
                  f"(offset={np.round(_grasp_state['offset'],3)})")
        if _grasp_state["holding"]:
            return (ee_pos + _grasp_state["offset"]).astype(np.float32)
        return ref
    else:  # 그리퍼 열림
        if _grasp_state["holding"]:
            _grasp_state["holding"] = False
            _grasp_state["resting"] = (ee_pos + _grasp_state["offset"]).astype(np.float32)  # 놓은 자리 고정
            print(f"[deploy] 놓음 → 큐브를 놓은 자리에 고정 "
                  f"(pos={np.round(_grasp_state['resting'],3)})")
        return _grasp_state["resting"] if _grasp_state["resting"] is not None else cam

def build_obs():
    wait_for_camera()                          # 카메라 값 없으면 여기서 대기(정답 폴백 안 함)
    joint_pos = np.array(franka.get_joint_positions(), dtype=np.float32)
    joint_vel = np.clip(franka.get_joint_velocities(), -50.0, 50.0).astype(np.float32)
    try:
        ee_pos, _ = franka.end_effector.get_world_pose()
        ee_pos = np.array(ee_pos, dtype=np.float32)
    except Exception:
        hand = stage.GetPrimAtPath("/World/Franka/panda_hand")
        xf = UsdGeom.Xformable(hand).ComputeLocalToWorldTransform(0)
        t = xf.ExtractTranslation()
        ee_pos = np.array([t[0], t[1], t[2]], dtype=np.float32)
    cube_world_pos = compute_cube_world_pos(joint_pos, ee_pos)   # 파지 전 카메라 / 파지 후 손 추적
    cube_rel   = cube_world_pos - ee_pos
    target_rel = target_pos     - ee_pos
    obs = np.concatenate([joint_pos, joint_vel, cube_rel, target_rel, cube_world_pos])
    return obs, joint_pos, cube_world_pos, cube_rel

def predict_chunk(obs):
    obs_norm = (obs - obs_mean) / obs_std
    with torch.no_grad():
        out = policy(torch.FloatTensor(obs_norm).unsqueeze(0)).squeeze(0).numpy()
    chunk = out.reshape(CHUNK_H, JOINT_DIM)
    chunk = chunk * act_std + act_mean

    GRIPPER_THRESHOLD = 0.035
    for h in range(CHUNK_H):
        if chunk[h, 7] < GRIPPER_THRESHOLD:
            chunk[h, 7:9] = 0.0
        else:
            chunk[h, 7:9] = 0.04
    return chunk


# ============================================================================
# 5. Receding-horizon chunk 제어 루프
# ============================================================================
print(f"\n[deploy] 신경망(chunk)으로 Franka 제어 시작!\n")

def run_tracking_demo():
    """(테스트) 1단계: 팔은 시작자세로 가만히 둔 채 큐브만 몇 번 랜덤 위치로 옮긴다.
    - 카메라 모드(USE_CAMERA_CUBE=True): 카메라 추정이 실제 위치를 따라오는지 오차 출력 + 일치 대기.
    - 정답 모드(USE_CAMERA_CUBE=False): 카메라 없이 물리만 잠깐 안정화(정책은 정답좌표 사용).
    끝나면 큐브는 마지막 랜덤 위치에 그대로 → 2단계(파지)."""
    if not MOVE_CUBE_BEFORE_GRASP:
        return
    use_cam = (receiver is not None)
    tag = "" if use_cam else "  [정답좌표 모드: 카메라 안 씀]"
    print(f"\n[test] === 1단계: 큐브 랜덤 이동 (팔 정지){tag} ===")

    def settle_on(target_xy):
        """카메라 모드면 추정이 target 근처(4cm)로 올 때까지, 정답 모드면 물리 안정화만 하고 통과."""
        est = None
        for k in range(SETTLE_MAX_STEPS):
            franka.apply_action(ArticulationAction(joint_positions=start_full))
            world.step(render=not HEADLESS)
            if use_cam:
                est = receiver.get_any()
                if est is not None and np.linalg.norm(est[:2] - target_xy) < 0.04:
                    return est
            elif k >= 60:
                return None
        return est

    for i in range(MAX_MOVES):
        if FIX_LAST_MOVE and i == MAX_MOVES - 1:
            newp = np.array(GRASP_POS, dtype=float)   # 마지막은 확실히 집히는 고정 위치
        else:
            newp = np.array([np.random.uniform(*MOVE_X), np.random.uniform(*MOVE_Y), 0.025], dtype=float)
        try:
            cube.set_world_pose(position=newp, orientation=np.array([1.0, 0.0, 0.0, 0.0]))
            cube.set_linear_velocity(np.zeros(3)); cube.set_angular_velocity(np.zeros(3))
        except Exception as e:
            print(f"[test] 큐브 이동 실패: {type(e).__name__}: {e}")
        est = settle_on(newp[:2])
        if not use_cam:
            print(f"[test] 이동 {i+1}/{MAX_MOVES}: 위치=({newp[0]:.3f},{newp[1]:.3f}) [정답좌표]")
        elif est is None:
            print(f"[test] 이동 {i+1}/{MAX_MOVES}: 실제=({newp[0]:.3f},{newp[1]:.3f}) → 카메라추정 아직 없음")
        else:
            gap = float(np.linalg.norm(est[:2] - newp[:2]))
            print(f"[test] 이동 {i+1}/{MAX_MOVES}: 실제=({newp[0]:.3f},{newp[1]:.3f}) "
                  f"카메라추정=({est[0]:.3f},{est[1]:.3f}) 오차={gap:.3f}m")
        # 옮긴 자리에 잠깐 멈춰서 GUI로 큐브가 이동한 걸 눈으로 볼 수 있게 함
        for _ in range(MOVE_DWELL_STEPS):
            franka.apply_action(ArticulationAction(joint_positions=start_full))
            world.step(render=not HEADLESS)

    # 마지막 랜덤 위치 그대로 고정 → (카메라 모드면) 추정이 그 위치에 맞을 때까지 대기
    try:
        gt, _ = cube.get_world_pose(); gt = np.array(gt, dtype=float)
    except Exception:
        gt = np.array(CUBE_POS, dtype=float)
    est = settle_on(gt[:2])
    if use_cam:
        gap = None if est is None else float(np.linalg.norm(est[:2] - gt[:2]))
        if gap is None or gap > 0.05:
            print(f"[test] ⚠ 카메라가 최종 위치를 못 잡음(오차={gap}). perception 이 느리거나 가려짐 → "
                  f"perception 을 detector:=color 로 재실행 권장. 이 상태면 팔이 엉뚱한 곳을 집습니다.")
    print(f"[test] 최종 랜덤 위치 ({gt[0]:.2f}, {gt[1]:.2f})에 큐브 고정")
    print("[test] === 1단계 끝 → 큐브 고정(랜덤 위치). 2단계: 파지 시작 ===\n")

run_tracking_demo()          # 1단계: 큐브 이동 + 카메라 추적 확인 (팔 정지). 끝나면 큐브 고정.
step = 0
while step < MAX_STEPS:
    obs, joint_pos, cube_world_pos, cube_rel = build_obs()
    chunk = predict_chunk(obs)

    cube_target_dist = np.linalg.norm(cube_world_pos[:2] - target_pos[:2])

    # 깔끔한 놓기: 큐브를 든 채(그리퍼 닫힘) 목표 XY 근처에 왔고, 큐브가 놓기 높이까지
    # 내려왔으면 → 이번 청크 동안 그리퍼를 '살짝 위'에서 열어 큐브를 밀치지 않게 한다.
    carrying = joint_pos[7] < GRIPPER_CLOSED_THRESH
    if (CLEAN_RELEASE and carrying
            and cube_target_dist < PLACE_XY_TOL
            and cube_world_pos[2] <= PLACE_RELEASE_Z):
        chunk[:, 7:9] = 0.04
        if step % 120 == 0:
            print(f"[deploy] 깔끔한 놓기: 목표 위 {cube_world_pos[2]:.3f}m 에서 그리퍼 열기")

    # 종료 조건(두 모드 모두):
    #  - 카메라 모드: 큐브를 집었다가(holding→resting) 그리퍼를 열어 '놓은' 순간
    #  - 정답 모드: 큐브가 목표 근처 + 그리퍼 열림 (원래 방식)
    # 놓는 즉시 끝내서, 과제 끝난 뒤 정책이 헤매는(지랄발광) 걸 막는다.
    placed = (_grasp_state["resting"] is not None) or (cube_target_dist < SUCCESS_XY_TOL)
    if placed and joint_pos[7] > 0.035:
        within = "성공" if cube_target_dist < SUCCESS_XY_TOL else "완료(오차 큼)"
        print(f"\n[deploy] 🎉 큐브 배치 {within}! (목표 오차: {cube_target_dist:.3f}m)")
        print("[deploy] 3초 대기 중...")
        for _ in range(180):
            world.step(render=not HEADLESS)
        print("[deploy] 초기 시작 자세로 복귀합니다...")
        for _ in range(120):
            franka.apply_action(ArticulationAction(joint_positions=start_full))
            world.step(render=not HEADLESS)
        print("[deploy] 복귀 완료. 프로그램을 깔끔하게 종료합니다.")
        break

    if step % 120 == 0 or step == 0:
        print(f"[deploy] step {step:4d} | 큐브(추정) {np.round(cube_world_pos,3)} "
              f"| 손→큐브 {np.linalg.norm(cube_rel):.3f}m")

    for h in range(min(REPLAN_EVERY, CHUNK_H)):
        if step >= MAX_STEPS:
            break
        franka.apply_action(ArticulationAction(joint_positions=chunk[h].astype(float)))
        world.step(render=not HEADLESS)
        step += 1

print("\n[deploy] 완료!")
simulation_app.close()
