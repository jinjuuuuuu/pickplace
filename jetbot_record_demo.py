# jetbot_record_demo.py  (RGB camera + /clock + /joint_states, auto-find robot root)
# ---------------------------------------------------------------------------
# Isaac Sim 5.1 standalone (Windows). Publishes:
#   /jetbot/camera/rgb, /clock, /joint_states
# mcap recording is done by rosbag2 on WSL2 (record_jetbot.launch.py).
# ---------------------------------------------------------------------------

# === User settings ==========================================================
SIM_SECONDS    = 60.0
HEADLESS       = False
CAMERA_TOPIC   = "/jetbot/camera/rgb"
CAMERA_FRAME   = "jetbot_camera"
LIN_VEL        = 0.20
ANG_VEL        = 0.60
WHEEL_RADIUS   = 0.0325
WHEEL_BASE     = 0.1125

ISAACSIM_PATH  = r"C:\isaacsim"
FASTDDS_XML    = r"C:\.ros\fastdds.xml"
JETBOT_USD     = r"C:\Users\user\Desktop\claude_jetbot\jetbot.usd"
# ============================================================================

import os, sys
if sys.platform == "win32":
    os.environ.setdefault("ROS_DISTRO", "humble")
    os.environ.setdefault("RMW_IMPLEMENTATION", "rmw_fastrtps_cpp")
    os.environ.setdefault("ROS_DOMAIN_ID", "0")
    if os.path.exists(FASTDDS_XML):
        os.environ.setdefault("FASTRTPS_DEFAULT_PROFILES_FILE", FASTDDS_XML)
    bridge_lib = os.path.join(ISAACSIM_PATH, "exts", "isaacsim.ros2.bridge", "humble", "lib")
    if os.path.isdir(bridge_lib):
        os.environ["PATH"] = bridge_lib + os.pathsep + os.environ.get("PATH", "")
        try:
            os.add_dll_directory(bridge_lib)
        except Exception:
            pass
        print(f"[demo] ROS2 bridge lib path added: {bridge_lib}")

from isaacsim import SimulationApp
simulation_app = SimulationApp({"renderer": "RayTracedLighting", "headless": HEADLESS})

import numpy as np
import omni.usd
import omni.graph.core as og
from pxr import Usd, UsdGeom, UsdPhysics, PhysxSchema

from isaacsim.core.api import World
from isaacsim.core.utils import extensions
from isaacsim.core.utils.nucleus import get_assets_root_path
from isaacsim.core.utils.stage import add_reference_to_stage
from isaacsim.core.nodes.scripts.utils import set_target_prims
from isaacsim.core.utils.types import ArticulationAction
from isaacsim.core.prims import SingleArticulation

extensions.enable_extension("isaacsim.ros2.bridge")
simulation_app.update()

world = World(stage_units_in_meters=1.0)
world.scene.add_default_ground_plane()

REF_PRIM = "/World/Jetbot"   # where we attach the reference
jetbot_usd = JETBOT_USD if JETBOT_USD else (get_assets_root_path() + "/Isaac/Robots/Jetbot/jetbot.usd")
add_reference_to_stage(usd_path=jetbot_usd, prim_path=REF_PRIM)
print(f"[demo] JetBot reference added: {jetbot_usd}")

print("[demo] waiting for asset to load...")
for _ in range(150):
    simulation_app.update()

stage = omni.usd.get_context().get_stage()

# --- find the ARTICULATION ROOT under the reference -------------------------
def find_articulation_root(root_path):
    root = stage.GetPrimAtPath(root_path)
    for p in Usd.PrimRange(root):
        if p.HasAPI(UsdPhysics.ArticulationRootAPI) or p.HasAPI(PhysxSchema.PhysxArticulationAPI):
            return p.GetPath().pathString
    return None

ROBOT_PRIM = find_articulation_root(REF_PRIM)
if ROBOT_PRIM is None:
    # fallback: the single child under the reference (e.g. /World/Jetbot/jetbot)
    children = stage.GetPrimAtPath(REF_PRIM).GetChildren()
    ROBOT_PRIM = children[0].GetPath().pathString if children else REF_PRIM
print(f"[demo] articulation root: {ROBOT_PRIM}")

# --- find the CAMERA (exclude viewport cameras) -----------------------------
def find_camera(root_path):
    cams = [p.GetPath().pathString for p in stage.Traverse()
            if p.IsA(UsdGeom.Camera) and not p.GetPath().pathString.startswith("/OmniverseKit_")]
    print(f"[demo] candidate cameras: {cams}")
    under = [c for c in cams if c.startswith(root_path)]
    return under[0] if under else (cams[0] if cams else None)

CAMERA_PRIM = find_camera(REF_PRIM)
if CAMERA_PRIM is None:
    print("[demo][ERROR] no camera found under the robot.")
    simulation_app.close()
    raise SystemExit(1)
print(f"[demo] using camera prim: {CAMERA_PRIM}")

# --- ROS 2 graph: RGB camera + /clock + /joint_states -----------------------
GRAPH_PATH = "/World/ROS_Camera"
og.Controller.edit(
    {"graph_path": GRAPH_PATH, "evaluator_name": "execution"},
    {
        og.Controller.Keys.CREATE_NODES: [
            ("OnTick",       "omni.graph.action.OnPlaybackTick"),
            ("Context",      "isaacsim.ros2.bridge.ROS2Context"),
            ("ReadSimTime",  "isaacsim.core.nodes.IsaacReadSimulationTime"),
            ("CreateRP",     "isaacsim.core.nodes.IsaacCreateRenderProduct"),
            ("CameraRgb",    "isaacsim.ros2.bridge.ROS2CameraHelper"),
            ("PublishClock", "isaacsim.ros2.bridge.ROS2PublishClock"),
            ("PublishJoint", "isaacsim.ros2.bridge.ROS2PublishJointState"),
        ],
        og.Controller.Keys.CONNECT: [
            ("OnTick.outputs:tick",                 "CreateRP.inputs:execIn"),
            ("CreateRP.outputs:execOut",            "CameraRgb.inputs:execIn"),
            ("CreateRP.outputs:renderProductPath",  "CameraRgb.inputs:renderProductPath"),
            ("Context.outputs:context",             "CameraRgb.inputs:context"),
            ("OnTick.outputs:tick",                 "PublishClock.inputs:execIn"),
            ("Context.outputs:context",             "PublishClock.inputs:context"),
            ("ReadSimTime.outputs:simulationTime",  "PublishClock.inputs:timeStamp"),
            ("OnTick.outputs:tick",                 "PublishJoint.inputs:execIn"),
            ("Context.outputs:context",             "PublishJoint.inputs:context"),
            ("ReadSimTime.outputs:simulationTime",  "PublishJoint.inputs:timeStamp"),
        ],
        og.Controller.Keys.SET_VALUES: [
            ("CameraRgb.inputs:frameId",      CAMERA_FRAME),
            ("CameraRgb.inputs:topicName",    CAMERA_TOPIC),
            ("CameraRgb.inputs:type",         "rgb"),
            ("PublishClock.inputs:topicName", "/clock"),
            ("PublishJoint.inputs:topicName", "/joint_states"),
        ],
    },
)
set_target_prims(primPath=f"{GRAPH_PATH}/CreateRP",
                 targetPrimPaths=[CAMERA_PRIM], inputName="inputs:cameraPrim")
set_target_prims(primPath=f"{GRAPH_PATH}/PublishJoint",
                 targetPrimPaths=[ROBOT_PRIM], inputName="inputs:targetPrim")
print(f"[demo] ROS2 graph built -> {CAMERA_TOPIC}, /clock, /joint_states")

# --- initialize & drive -----------------------------------------------------
world.reset()
jetbot = SingleArticulation(prim_path=ROBOT_PRIM, name="jetbot")
jetbot.initialize()
print(f"[demo] JetBot DOF names: {jetbot.dof_names}")

wheel_idx = [i for i, n in enumerate(jetbot.dof_names) if "wheel" in n.lower()][:2]
print(f"[demo] wheel joints: {[jetbot.dof_names[i] for i in wheel_idx]}")

v_l = (LIN_VEL - ANG_VEL * WHEEL_BASE / 2.0) / WHEEL_RADIUS
v_r = (LIN_VEL + ANG_VEL * WHEEL_BASE / 2.0) / WHEEL_RADIUS

print(f"[demo] running for {SIM_SECONDS}s (WSL2 must be recording).")
elapsed, dt = 0.0, 1.0 / 60.0
while simulation_app.is_running() and elapsed < SIM_SECONDS:
    if len(wheel_idx) >= 2:
        vel = np.zeros(jetbot.num_dof)
        vel[wheel_idx[0]] = v_l
        vel[wheel_idx[1]] = v_r
        jetbot.apply_action(ArticulationAction(joint_velocities=vel))
    world.step(render=not HEADLESS)
    elapsed += dt

print("[demo] done. closing.")
simulation_app.close()