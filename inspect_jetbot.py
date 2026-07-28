# inspect_jetbot.py
# ---------------------------------------------------------------------------
# Diagnostic for Isaac Sim 5.1 (Windows), using YOUR camera-equipped JetBot.
#   1) Prints every prim under the JetBot -> find the real camera path
#   2) Prints available OmniGraph node types (ROS2 bridge / core) -> correct names
# Run:
#   python.bat "C:\Users\user\Desktop\claude_jetbot\inspect_jetbot.py"
# ---------------------------------------------------------------------------

ISAACSIM_PATH = r"C:\isaacsim"
JETBOT_USD    = r"C:\Users\user\Desktop\claude_jetbot\jetbot_camera.usd"   # your file

import os, sys
if sys.platform == "win32":
    os.environ.setdefault("ROS_DISTRO", "humble")
    os.environ.setdefault("RMW_IMPLEMENTATION", "rmw_fastrtps_cpp")
    bridge_lib = os.path.join(ISAACSIM_PATH, "exts", "isaacsim.ros2.bridge", "humble", "lib")
    if os.path.isdir(bridge_lib):
        os.environ["PATH"] = bridge_lib + os.pathsep + os.environ.get("PATH", "")
        try:
            os.add_dll_directory(bridge_lib)
        except Exception:
            pass

from isaacsim import SimulationApp
simulation_app = SimulationApp({"headless": True})

import omni.usd
import omni.graph.core as og
from pxr import Usd, UsdGeom

from isaacsim.core.api import World
from isaacsim.core.utils import extensions
from isaacsim.core.utils.stage import add_reference_to_stage

extensions.enable_extension("isaacsim.ros2.bridge")
simulation_app.update()

world = World(stage_units_in_meters=1.0)
world.scene.add_default_ground_plane()

JETBOT_PRIM = "/World/Jetbot"
if not os.path.exists(JETBOT_USD):
    print(f"[inspect][ERROR] file not found: {JETBOT_USD}")
    simulation_app.close()
    raise SystemExit
add_reference_to_stage(usd_path=JETBOT_USD, prim_path=JETBOT_PRIM)

print("[inspect] loading...")
for _ in range(150):
    simulation_app.update()

stage = omni.usd.get_context().get_stage()

print("\n===== (1) FULL JETBOT PRIM TREE =====")
root = stage.GetPrimAtPath(JETBOT_PRIM)
cam_paths = []
for p in Usd.PrimRange(root):
    t = p.GetTypeName()
    is_cam = p.IsA(UsdGeom.Camera)
    if is_cam:
        cam_paths.append(p.GetPath().pathString)
    mark = "  <-- CAMERA" if is_cam else ""
    print(f"  {p.GetPath().pathString}   [{t}]{mark}")

print("\n===== CAMERAS UNDER JETBOT =====")
for c in cam_paths:
    print("  ", c)
if not cam_paths:
    print("  (none found under /World/Jetbot - camera may be outside the robot prim)")

print("\n===== (2) RELEVANT OG NODE TYPES =====")
try:
    all_types = og.get_node_type_names() if hasattr(og, "get_node_type_names") else []
except Exception:
    all_types = []
keys = ("PlaybackTick", "ROS2Context", "RenderProduct", "CameraHelper", "CameraInfo",
        "ReadSimulationTime", "PublishClock", "RunOneSimulationFrame", "SimulationGate")
hits = [t for t in sorted(set(all_types)) if any(k in t for k in keys)]
for t in hits:
    print("  ", t)
if not hits:
    print("  (could not enumerate; will rely on docs)")

print("\n[inspect] done.")
simulation_app.close()
