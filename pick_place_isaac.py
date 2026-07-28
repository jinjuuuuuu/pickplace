# pick_place_isaac.py  (Isaac Sim 5.1, Windows standalone)
# ---------------------------------------------------------------------------
# Loads Franka Panda and runs pick-and-place using:
#   * EXECUTION LOGIC  -> taken from panda_controller.py: a simple time-based
#                         SEQUENCE of (pose, gripper, delay) stepped by a timer,
#                         issued as direct joint commands. No built-in
#                         controller, no feedback state machine.
#   * SCENE DETAILS    -> kept from the original pick_place_isaac.py: cube /
#                         marker positions, the pose values (where the arm
#                         moves), the cameras, and the ROS2 topic recording.
#
# The cube is STATIC at CUBE_INIT_POS - nothing is teleported at runtime.
#
# Run:
#   "C:\isaacsim\python.bat" "C:\Users\user\Desktop\claude_jetbot\pick_place_isaac.py"
# (WSL2 must already be recording via record_pick_place.launch.py)
# ---------------------------------------------------------------------------

# === User settings ==========================================================
SIM_SECONDS     = 120.0
HEADLESS        = False
ISAACSIM_PATH   = r"C:\isaacsim"
FASTDDS_XML     = r"C:\.ros\fastdds.xml"

# Scene geometry (meters; Franka base is at world origin)
CUBE_INIT_POS   = [0.45,  0.00, 0.025]   # cube centre when resting on ground
PLACE_TARGET    = [0.45,  0.28, 0.025]   # visual marker for the place location
CUBE_HALF_SIZE  = 0.025                  # 5 cm cube (full edge = 0.05 m)

# Gripper drive gains (FIX 1)
# NOTE: the original 1e5 / 1e3 / 200 N values were WAY too aggressive. The wrist
# joints (panda_joint5/6/7) have a drive torque limit of only 12 N.m, so a 200 N
# grip force back-drives the wrist and makes the arm spin at the grasp instant.
# Keep grip force modest (~40 N is plenty for a 0.1 kg cube) and the drive scale
# close to the asset's native finger drive (stiffness 400, maxForce ~7 N).
FINGER_STIFFNESS = 2.0e3
FINGER_DAMPING   = 1.0e2
FINGER_MAX_FORCE = 40.0
FINGER_OPEN_POS  = 0.04                  # each finger fully open (m)
FINGER_CLOSED_POS = 0.0                  # each finger fully closed (m)

# Wrist camera (attached to panda_hand)
WRIST_RGB_TOPIC  = "/franka/wrist_camera/rgb"
WRIST_DEPTH_TOPIC= "/franka/wrist_camera/depth"
WRIST_INFO_TOPIC = "/franka/wrist_camera/camera_info"
WRIST_CAM_FRAME  = "panda_hand_camera"

# Overhead camera (fixed, looking down at scene)
OVERHEAD_RGB_TOPIC = "/overhead_camera/rgb"
OVERHEAD_CAM_FRAME = "overhead_camera"
OVERHEAD_CAM_POS   = [0.4, 0.15, 1.5]   # (x, y, z) above scene
OVERHEAD_CAM_ROT   = (0, 0, 0)           # Euler XYZ deg. In Isaac (Z-up) a camera
                                         # at rotation 0 looks straight DOWN (-Z).
                                         # The old (0,90,0) made it look sideways,
                                         # grazing the floor -> only grid lines.
# =============================================================================

import os, sys
import numpy as np

# --- ROS2 / DDS env (Windows) ------------------------------------------------
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
        print(f"[demo] ROS2 bridge lib added: {bridge_lib}")

# --- Isaac Sim app -----------------------------------------------------------
from isaacsim import SimulationApp
simulation_app = SimulationApp({"renderer": "RayTracedLighting", "headless": HEADLESS})

# deferred imports (must come after SimulationApp)
import omni.usd
import omni.graph.core as og
from pxr import Usd, UsdGeom, UsdPhysics, PhysxSchema, Gf, Sdf

from isaacsim.core.api   import World
from isaacsim.core.api.objects import DynamicCuboid
from isaacsim.core.utils import extensions
from isaacsim.core.utils.nucleus    import get_assets_root_path
from isaacsim.core.utils.stage      import add_reference_to_stage
from isaacsim.core.nodes.scripts.utils import set_target_prims
from isaacsim.core.prims            import SingleArticulation
from isaacsim.core.utils.types      import ArticulationAction

extensions.enable_extension("isaacsim.ros2.bridge")
simulation_app.update()

# ============================================================================
# Scene setup
# ============================================================================
world = World(stage_units_in_meters=1.0)
world.scene.add_default_ground_plane()

stage = omni.usd.get_context().get_stage()

# --- Franka Panda: local file first, fall back to NVIDIA cloud --------------
_LOCAL_FRANKA = r"C:\Users\user\Desktop\claude_jetbot\franka\franka.usd"
if os.path.exists(_LOCAL_FRANKA):
    FRANKA_USD = _LOCAL_FRANKA
    print(f"[demo] Using local Franka USD: {FRANKA_USD}")
else:
    FRANKA_USD = get_assets_root_path() + "/Isaac/Robots/Franka/franka.usd"
    print(f"[demo] Local file not found, using cloud: {FRANKA_USD}")
ROBOT_REF    = "/World/Franka"

add_reference_to_stage(usd_path=FRANKA_USD, prim_path=ROBOT_REF)
print(f"[demo] Franka reference added: {FRANKA_USD}")

# --- Cube (rigid body) -------------------------------------------------------
CUBE_PRIM_PATH = "/World/PickCube"
cube = world.scene.add(DynamicCuboid(
    prim_path = CUBE_PRIM_PATH,
    name      = "pick_cube",
    position  = np.array(CUBE_INIT_POS),
    size      = CUBE_HALF_SIZE * 2.0,     # full edge length
    color     = np.array([0.8, 0.2, 0.1]),
    mass      = 0.1,
))
print(f"[demo] Cube added at {CUBE_INIT_POS}")

# --- Target marker (visual only, no physics) ---------------------------------
MARKER_PRIM_PATH = "/World/PlaceMarker"
marker = stage.DefinePrim(MARKER_PRIM_PATH, "Cylinder")
UsdGeom.Cylinder(marker).GetRadiusAttr().Set(CUBE_HALF_SIZE)
UsdGeom.Cylinder(marker).GetHeightAttr().Set(0.002)
UsdGeom.XformCommonAPI(marker).SetTranslate(Gf.Vec3d(*PLACE_TARGET))
UsdGeom.Gprim(marker).GetDisplayColorAttr().Set([Gf.Vec3f(0.1, 0.8, 0.1)])

# --- Load frames & wait for Nucleus ------------------------------------------
print("[demo] waiting for assets to load ...")
MAX_WAIT_FRAMES = 3000   # ~50 s at 60 fps; increase if on slow connection
for i in range(MAX_WAIT_FRAMES):
    simulation_app.update()
    stage = omni.usd.get_context().get_stage()
    ref_prim = stage.GetPrimAtPath(ROBOT_REF)
    if ref_prim.IsValid() and len(list(ref_prim.GetChildren())) > 0:
        print(f"[demo] Franka asset loaded after {i} frames (~{i/60:.1f}s)")
        break
    if i % 120 == 0 and i > 0:
        print(f"[demo] still waiting for Franka... ({i/60:.0f}s elapsed)")
else:
    print("[demo][ERROR] Franka failed to load within timeout. Check Nucleus/network.")
    simulation_app.close()
    raise SystemExit(1)

# ============================================================================
# Print prim tree and discover articulation root
# ============================================================================
print("[demo] Prim tree under Franka reference:")
ref_root = stage.GetPrimAtPath(ROBOT_REF)
for idx, p in enumerate(Usd.PrimRange(ref_root)):
    depth  = len(p.GetPath().pathString.split("/")) - len(ROBOT_REF.split("/"))
    indent = "  " * depth
    has_art = (p.HasAPI(UsdPhysics.ArticulationRootAPI) or
               p.HasAPI(PhysxSchema.PhysxArticulationAPI))
    tag = " [ARTICULATION ROOT]" if has_art else ""
    print(f"  {indent}{p.GetPath().pathString}  [{p.GetTypeName()}]{tag}")
    if idx > 40:
        print("  ... (truncated)")
        break

def find_articulation_root(root_path):
    """Return the first prim (depth-first) with ArticulationRootAPI."""
    root = stage.GetPrimAtPath(root_path)
    for p in Usd.PrimRange(root):
        if p.GetPath().pathString == root_path:
            continue
        if (p.HasAPI(UsdPhysics.ArticulationRootAPI) or
                p.HasAPI(PhysxSchema.PhysxArticulationAPI)):
            return p.GetPath().pathString
    if (root.HasAPI(UsdPhysics.ArticulationRootAPI) or
            root.HasAPI(PhysxSchema.PhysxArticulationAPI)):
        return root_path
    children = list(root.GetChildren())
    return children[0].GetPath().pathString if children else root_path

ROBOT_PRIM = find_articulation_root(ROBOT_REF)
print(f"[demo] articulation root: {ROBOT_PRIM}")

# ============================================================================
# Attach wrist camera to panda_hand (or first suitable EE link)
# ============================================================================
def find_ee_prim(root_path, candidates=("panda_hand", "panda_link8", "flange")):
    ref_root = stage.GetPrimAtPath(root_path)
    for prim in Usd.PrimRange(ref_root):
        if prim.GetName() in candidates:
            print(f"[demo] found EE candidate: {prim.GetPath().pathString}")
            return prim.GetPath().pathString
    print(f"[demo] EE prim not found; falling back to robot root: {root_path}")
    return root_path

EE_PRIM = find_ee_prim(ROBOT_REF)
print(f"[demo] end-effector prim: {EE_PRIM}")

WRIST_CAM_PRIM = EE_PRIM + "/WristCamera"
cam_xform = stage.DefinePrim(WRIST_CAM_PRIM, "Xform")
WRIST_CAM_USD  = WRIST_CAM_PRIM + "/Camera"
wrist_cam = stage.DefinePrim(WRIST_CAM_USD, "Camera")

# [FIX 6] A bare Camera prim defaults to near-clip = 1.0 m and a narrow 50 mm
# lens. In a metre-scale scene that clips everything closer than 1 m -> black
# image (the wrist cam is only ~0.1 m from the cube). Give every camera a small
# near clip and a wider lens.
def _setup_cam(prim, focal_mm, near=0.01, far=1.0e5):
    c = UsdGeom.Camera(prim)
    c.CreateClippingRangeAttr(Gf.Vec2f(float(near), float(far)))
    c.CreateFocalLengthAttr(float(focal_mm))
    c.CreateHorizontalApertureAttr(20.955)
    c.CreateVerticalApertureAttr(20.955 * 480.0 / 640.0)   # 4:3 to match 640x480
_setup_cam(wrist_cam, focal_mm=16.0)   # wide for close-up eye-in-hand view
UsdGeom.XformCommonAPI(cam_xform).SetTranslate(Gf.Vec3d(0.0, 0.0, 0.05))
# panda_hand's local +Z is the gripper APPROACH axis (points toward the fingers /
# the object being grasped). A USD camera looks down its local -Z, so rotate 180
# deg about X to make the camera look along +Z -> an eye-in-hand view of the cube.
# (The old (0,-60,0) pointed it into empty space -> all black.)
UsdGeom.XformCommonAPI(cam_xform).SetRotate(Gf.Vec3f(180, 0, 0))
print(f"[demo] wrist camera prim: {WRIST_CAM_USD}")

# --- Overhead camera (fixed) -------------------------------------------------
OVERHEAD_XFORM = "/World/OverheadCamXform"
overhead_xform = stage.DefinePrim(OVERHEAD_XFORM, "Xform")
OVERHEAD_CAM_USD = OVERHEAD_XFORM + "/Camera"
overhead_cam = stage.DefinePrim(OVERHEAD_CAM_USD, "Camera")
_setup_cam(overhead_cam, focal_mm=24.0)   # ~47 deg FOV: sees the whole workspace
UsdGeom.XformCommonAPI(overhead_xform).SetTranslate(Gf.Vec3d(*OVERHEAD_CAM_POS))
UsdGeom.XformCommonAPI(overhead_xform).SetRotate(Gf.Vec3f(*OVERHEAD_CAM_ROT))
print(f"[demo] overhead camera prim: {OVERHEAD_CAM_USD}")

for _ in range(50):
    simulation_app.update()

# ============================================================================
# OmniGraph: ROS2 publishers   [KEEP - topic recording, unchanged]
# ============================================================================
GRAPH_PATH = "/World/ROS_PnP"

og.Controller.edit(
    {"graph_path": GRAPH_PATH, "evaluator_name": "execution"},
    {
        og.Controller.Keys.CREATE_NODES: [
            ("OnTick",           "omni.graph.action.OnPlaybackTick"),
            ("Context",          "isaacsim.ros2.bridge.ROS2Context"),
            ("ReadSimTime",      "isaacsim.core.nodes.IsaacReadSimulationTime"),
            ("CreateRP_Wrist",   "isaacsim.core.nodes.IsaacCreateRenderProduct"),
            ("WristRGB",         "isaacsim.ros2.bridge.ROS2CameraHelper"),
            ("WristDepth",       "isaacsim.ros2.bridge.ROS2CameraHelper"),
            ("CreateRP_Over",    "isaacsim.core.nodes.IsaacCreateRenderProduct"),
            ("OverheadRGB",      "isaacsim.ros2.bridge.ROS2CameraHelper"),
            ("PublishClock",     "isaacsim.ros2.bridge.ROS2PublishClock"),
            ("PublishJoint",     "isaacsim.ros2.bridge.ROS2PublishJointState"),
            ("PublishTF",        "isaacsim.ros2.bridge.ROS2PublishTransformTree"),
        ],
        og.Controller.Keys.CONNECT: [
            ("OnTick.outputs:tick",                  "CreateRP_Wrist.inputs:execIn"),
            ("CreateRP_Wrist.outputs:execOut",           "WristRGB.inputs:execIn"),
            ("CreateRP_Wrist.outputs:execOut",           "WristDepth.inputs:execIn"),
            ("CreateRP_Wrist.outputs:renderProductPath", "WristRGB.inputs:renderProductPath"),
            ("CreateRP_Wrist.outputs:renderProductPath", "WristDepth.inputs:renderProductPath"),
            ("Context.outputs:context",                  "WristRGB.inputs:context"),
            ("Context.outputs:context",                  "WristDepth.inputs:context"),

            ("OnTick.outputs:tick",                  "CreateRP_Over.inputs:execIn"),
            ("CreateRP_Over.outputs:execOut",         "OverheadRGB.inputs:execIn"),
            ("CreateRP_Over.outputs:renderProductPath","OverheadRGB.inputs:renderProductPath"),
            ("Context.outputs:context",              "OverheadRGB.inputs:context"),

            ("OnTick.outputs:tick",                  "PublishClock.inputs:execIn"),
            ("Context.outputs:context",              "PublishClock.inputs:context"),
            ("ReadSimTime.outputs:simulationTime",   "PublishClock.inputs:timeStamp"),

            ("OnTick.outputs:tick",                  "PublishJoint.inputs:execIn"),
            ("Context.outputs:context",              "PublishJoint.inputs:context"),
            ("ReadSimTime.outputs:simulationTime",   "PublishJoint.inputs:timeStamp"),

            ("OnTick.outputs:tick",                  "PublishTF.inputs:execIn"),
            ("Context.outputs:context",              "PublishTF.inputs:context"),
            ("ReadSimTime.outputs:simulationTime",   "PublishTF.inputs:timeStamp"),
        ],
        og.Controller.Keys.SET_VALUES: [
            # [FIX 4] Render products MUST have an explicit resolution, otherwise
            # they produce no frame -> ROS2CameraHelper advertises the topic but
            # publishes 0 images (exactly the symptom: all camera topics Count=0
            # while clock/joint/tf record fine).
            ("CreateRP_Wrist.inputs:width",  640),
            ("CreateRP_Wrist.inputs:height", 480),
            ("CreateRP_Wrist.inputs:enabled", True),
            ("CreateRP_Over.inputs:width",   640),
            ("CreateRP_Over.inputs:height",  480),
            ("CreateRP_Over.inputs:enabled", True),
            ("WristRGB.inputs:topicName",   WRIST_RGB_TOPIC),
            ("WristRGB.inputs:frameId",     WRIST_CAM_FRAME),
            ("WristRGB.inputs:type",        "rgb"),
            ("WristDepth.inputs:topicName", WRIST_DEPTH_TOPIC),
            ("WristDepth.inputs:frameId",   WRIST_CAM_FRAME),
            ("WristDepth.inputs:type",      "depth"),
            ("OverheadRGB.inputs:topicName",  OVERHEAD_RGB_TOPIC),
            ("OverheadRGB.inputs:frameId",    OVERHEAD_CAM_FRAME),
            ("OverheadRGB.inputs:type",       "rgb"),
            ("PublishClock.inputs:topicName", "/clock"),
            ("PublishJoint.inputs:topicName", "/joint_states"),
            ("PublishTF.inputs:topicName",    "/tf"),
        ],
    },
)

set_target_prims(primPath=f"{GRAPH_PATH}/CreateRP_Wrist",
                 targetPrimPaths=[WRIST_CAM_USD], inputName="inputs:cameraPrim")
set_target_prims(primPath=f"{GRAPH_PATH}/CreateRP_Over",
                 targetPrimPaths=[OVERHEAD_CAM_USD], inputName="inputs:cameraPrim")
set_target_prims(primPath=f"{GRAPH_PATH}/PublishJoint",
                 targetPrimPaths=[ROBOT_PRIM], inputName="inputs:targetPrim")
set_target_prims(primPath=f"{GRAPH_PATH}/PublishTF",
                 targetPrimPaths=[ROBOT_PRIM], inputName="inputs:targetPrims")

print("[demo] OmniGraph ROS2 publishers built.")

# ============================================================================
# [FIX 1] Ensure both finger joints have an active position drive
# ----------------------------------------------------------------------------
# panda_finger_joint2 is authored as a mimic of joint1, but the mimic coupling
# is ignored at runtime, so joint2 has no drive and never moves -> only one
# finger closes -> cube is never grasped. Apply an explicit linear drive to
# BOTH finger joints. Must run BEFORE world.reset().
# ============================================================================
def ensure_finger_drives(stiffness, damping, max_force, open_pos):
    """Apply a linear PD drive to panda_finger_joint1 and panda_finger_joint2 only."""
    JOINT_TYPES = {
        "PhysicsJoint", "PhysicsPrismaticJoint", "PhysicsRevoluteJoint",
        "PhysicsFixedJoint", "PhysicsSphericalJoint", "PhysicsD6Joint",
    }
    TARGET_NAMES = {"panda_finger_joint1", "panda_finger_joint2"}

    found = {}
    for p in Usd.PrimRange(stage.GetPrimAtPath(ROBOT_REF)):
        if p.GetName() not in TARGET_NAMES:
            continue
        type_name = p.GetTypeName()
        schemas = set(p.GetAppliedSchemas())
        has_joint_schema = any("joint" in s.lower() for s in schemas)
        if type_name in JOINT_TYPES or has_joint_schema or type_name == "":
            found[p.GetName()] = p
            print(f"[fix] finger joint found: {p.GetPath().pathString}  [{type_name}]")

    for jname in TARGET_NAMES:
        jprim = found.get(jname)
        if jprim is None:
            print(f"[fix][WARN] {jname} not found - drive NOT applied")
            continue
        drive = UsdPhysics.DriveAPI.Apply(jprim, "linear")
        drive.CreateTypeAttr().Set("force")
        drive.CreateStiffnessAttr().Set(stiffness)
        drive.CreateDampingAttr().Set(damping)
        drive.CreateMaxForceAttr().Set(max_force)
        drive.CreateTargetPositionAttr().Set(open_pos)
        print(f"[fix] linear drive applied to {jprim.GetPath().pathString}")

ensure_finger_drives(FINGER_STIFFNESS, FINGER_DAMPING, FINGER_MAX_FORCE, FINGER_OPEN_POS)

# ============================================================================
# [FIX 2] Anchor the robot base to the world (fix the floating-base bug)
# ----------------------------------------------------------------------------
# This asset's "rootJoint" is a generic PhysicsJoint with the bodies REVERSED
# (body0 = panda_link0, body1 = <empty/world>) instead of the standard
# FixedJoint (body0 = world, body1 = panda_link0). PhysX therefore treats the
# articulation as a FLOATING base: when the arm drives apply torque, the whole
# robot tumbles/spins in mid-air and can never reach the cube.
# Reconfigure rootJoint as a proper world-anchored FixedJoint. Must run BEFORE
# world.reset() so PhysX parses the corrected joint.
# ============================================================================
def ensure_fixed_base():
    ref_root = stage.GetPrimAtPath(ROBOT_REF)
    root_joint = None
    for p in Usd.PrimRange(ref_root):
        if p.GetName() == "rootJoint":
            root_joint = p
            break
    if root_joint is None:
        print("[fix][WARN] rootJoint not found - base may stay floating!")
        return
    rel_b0 = root_joint.GetRelationship("physics:body0")
    rel_b1 = root_joint.GetRelationship("physics:body1")
    b0 = list(rel_b0.GetTargets()) if rel_b0 else []
    b1 = list(rel_b1.GetTargets()) if rel_b1 else []
    # The root link is whichever side currently references a body.
    base_link = b0[0] if b0 else (b1[0] if b1 else None)

    # Standard fixed-base pattern: body0 = world (empty), body1 = root link.
    root_joint.SetTypeName("PhysicsFixedJoint")
    if rel_b0:
        rel_b0.SetTargets([])              # empty -> world frame
    if rel_b1 and base_link is not None:
        rel_b1.SetTargets([base_link])     # child -> root link
    en = root_joint.GetAttribute("physics:jointEnabled")
    if en:
        en.Set(True)
    print(f"[fix] base anchored to world via FixedJoint (root link = {base_link})")

ensure_fixed_base()

# ============================================================================
# Initialize world & articulation
# ============================================================================
franka = world.scene.add(SingleArticulation(prim_path=ROBOT_PRIM, name="franka_arm"))
world.reset()
franka.initialize()
print(f"[demo] Franka DOF names: {franka.dof_names}")
n_dof = franka.num_dof
print(f"[demo] DOF count: {n_dof}")

# ----------------------------------------------------------------------------
# Map joint NAMES -> DOF indices, so the 7 arm angles always land on
# panda_joint1..7 in order regardless of the articulation's DOF ordering.
# ----------------------------------------------------------------------------
ARM_JOINT_NAMES    = ['panda_joint1', 'panda_joint2', 'panda_joint3', 'panda_joint4',
                      'panda_joint5', 'panda_joint6', 'panda_joint7']
FINGER_JOINT_NAMES = ['panda_finger_joint1', 'panda_finger_joint2']

name_to_idx = {n: i for i, n in enumerate(franka.dof_names)}
arm_order_idx    = [name_to_idx[n] for n in ARM_JOINT_NAMES    if n in name_to_idx]
finger_order_idx = [name_to_idx[n] for n in FINGER_JOINT_NAMES if n in name_to_idx]
print(f"[demo] arm joints (ordered):    {[franka.dof_names[i] for i in arm_order_idx]}")
print(f"[demo] finger joints (ordered): {[franka.dof_names[i] for i in finger_order_idx]}")
if len(arm_order_idx) != 7:
    print(f"[demo][WARN] expected 7 arm joints, found {len(arm_order_idx)} - "
          f"check joint names vs ARM_JOINT_NAMES.")

# ============================================================================
# Pick-and-place definition
#   LOGIC  : panda_controller.py-style time-based SEQUENCE (pose, gripper, delay)
#   VALUES : the arm poses from the original pick_place_isaac.py (unchanged)
# ============================================================================
# [FIX 3] The original joint angles did NOT reach the cube: FK showed the
# "grasp" pose put the gripper grasp point at [0.647, 0, 0.335] -- ~30 cm too
# high and ~20 cm too far forward, so the arm never descended near the cube.
# These values were recomputed by inverse kinematics (verified by FK to <1 mm)
# so the gripper grasp point (TCP, 0.1034 m below panda_hand) lands exactly on:
#   pre_grasp/lift : above the cube,  approach pointing straight down (-Z)
#   grasp          : [0.45, 0.00, 0.035] -> straddling the cube centre (z=0.025)
#   place_pre/place: above / at PLACE_TARGET [0.45, 0.28, 0.025]
POSES = {
    "home":      [0.0, -0.785, 0.0, -2.356, 0.0, 1.571, 0.785],
    "pre_grasp": [0.1393,  0.0067, -0.1391, -2.5623,  0.0017, 2.5690, 0.9084],
    "grasp":     [0.1509,  0.3699, -0.1175, -2.5314,  0.1748, 2.8954, 0.6979],
    "lift":      [0.0949, -0.2334, -0.0931, -2.4598, -0.0271, 2.2272, 0.9432],
    "place_pre": [0.5258,  0.2011,  0.0305, -2.2906, -0.0101, 2.4916, 0.6664],
    "place":     [0.3955,  0.5027,  0.1396, -2.2624, -0.1789, 2.7554, 0.3325],
}

# (pose_name, gripper_state, hold_seconds) - same choreography style as the ROS2 node.
SEQUENCE = [
    ("home",      "open",   3.0),
    ("pre_grasp", "open",   2.0),
    ("grasp",     "open",   1.5),
    ("grasp",     "closed", 1.5),   # close on the cube
    ("lift",      "closed", 1.5),
    ("place_pre", "closed", 2.0),
    ("place",     "closed", 1.5),
    ("place",     "open",   1.5),   # release
    ("place_pre", "open",   1.0),
    ("home",      "open",   2.0),
]

def gripper_value(state):
    return FINGER_OPEN_POS if state == "open" else FINGER_CLOSED_POS

def command_pose(arm_angles, gripper_pos):
    """Send one ArticulationAction: 7 arm joints + both fingers (by index)."""
    idx = arm_order_idx + finger_order_idx
    pos = list(arm_angles) + [gripper_pos] * len(finger_order_idx)
    franka.apply_action(ArticulationAction(
        joint_positions=np.array(pos, dtype=float),
        joint_indices=np.array(idx, dtype=int),
    ))

def finger_positions():
    jp = franka.get_joint_positions()
    if jp is None or not finger_order_idx:
        return None
    return [round(float(jp[i]), 4) for i in finger_order_idx]

# ---- Warm-up: open the gripper before the sequence -------------------------
# world.reset() sets fingers from the USD authored position (0.0 = closed);
# the drive target needs a few steps to push them open.
print("[demo] warm-up: opening gripper ...")
for _ in range(90):
    if finger_order_idx:
        franka.apply_action(ArticulationAction(
            joint_positions=np.full(len(finger_order_idx), FINGER_OPEN_POS),
            joint_indices=np.array(finger_order_idx, dtype=int),
        ))
    world.step(render=not HEADLESS)
print(f"[demo] warm-up done. finger pos: {finger_positions()}")

# ============================================================================
# Main simulation loop - run the SEQUENCE (time-based, like panda_controller.py)
# ============================================================================
print(f"[demo] Starting simulation for {SIM_SECONDS}s ...")
print("[demo] Ensure WSL2 is recording: ros2 launch record_pick_place.launch.py")

dt          = 1.0 / 60.0
elapsed     = 0.0
step        = 0
seq_idx     = 0
state_timer = 0.0          # seconds spent in the current sequence step
completed   = False

try:
    while simulation_app.is_running() and elapsed < SIM_SECONDS:

        if seq_idx < len(SEQUENCE):
            pose_name, gripper_state, hold_s = SEQUENCE[seq_idx]
            command_pose(POSES[pose_name], gripper_value(gripper_state))

            state_timer += dt
            if state_timer >= hold_s:
                print(f"[demo] -> step {seq_idx + 1}/{len(SEQUENCE)} "
                      f"'{pose_name}' ({gripper_state}) done (t={elapsed:.1f}s)")
                seq_idx     += 1
                state_timer  = 0.0
        elif not completed:
            completed = True
            print(f"[demo] Pick-and-place complete at t={elapsed:.1f}s "
                  f"(holding home until {SIM_SECONDS:.0f}s for recording).")
        else:
            command_pose(POSES["home"], FINGER_OPEN_POS)   # hold home

        world.step(render=not HEADLESS)
        elapsed += dt
        step    += 1

        # Log cube pose + finger positions every ~2 s
        if step % 120 == 0:
            try:
                c_pos, _ = cube.get_world_pose()
                print(f"[demo] t={elapsed:6.1f}s | cube pos: {np.round(np.asarray(c_pos), 3)} "
                      f"| fingers: {finger_positions()}")
            except Exception:
                pass

except Exception as e:
    import traceback
    print(f"[demo][ERROR] {e}")
    traceback.print_exc()
finally:
    print("[demo] Simulation finished. Closing.")
    simulation_app.close()