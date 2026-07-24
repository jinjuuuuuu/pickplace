# pick_place_collect.py  (Isaac Sim 5.1, Windows standalone)
# ---------------------------------------------------------------------------
# BC(모방학습) 데이터 수집 스크립트 v8 (큐브 위치 격자 샘플링으로 변경)
#
# v7까지는 큐브 위치를 CUBE_X_RANGE x CUBE_Y_RANGE 전체에 걸쳐 매 에피소드
# random.uniform으로 연속적으로 흩뿌렸다. 그 결과 학습된 정책이 "본 적 있는"
# 위치는 5mm 이내로 정확히 도달하지만 "새로운" 위치는 최대 20cm씩 빗나가는
# 일반화 실패가 확인됐다 (동일 지점 재현 시 성공, 임의 신규 지점은 실패).
#
# v8은 큐브 위치를 CUBE_GRID_NX x CUBE_GRID_NY 격자점으로 고정하고, 각 격자점을
# EPISODES_PER_POINT번 반복 수집한다 (지점당 데이터 밀도 확보가 핵심 — 총량을
# 넓은 연속 공간에 얇게 뿌리는 것보다, 몇 개 지점에 데이터를 집중시키는 쪽이
# 실제로 일반화가 된 사례들의 공통점이었다). 타겟 위치는 기존처럼 연속 랜덤
# 유지 (배치 지점 일반화는 아직 실패가 확인되지 않았고, 조합 폭발을 피하기 위함).
# === 설정 ===================================================================
MAX_STEPS       = 2000
HEADLESS        = True
RENDER          = True
ISAACSIM_PATH   = r"C:\isaacsim"
SAVE_PATH       = r"/data/jinju/bc_data_v5"

FRANKA_USD      = r"C:\Users\user\Desktop\isaacsim\isaac-sim-assets-robots_and_sensors-5.1.0\Assets\Isaac\5.1\Isaac\Robots\FrankaRobotics\FrankaPanda\franka.usd"

CUBE_X_RANGE    = (0.30, 0.55); CUBE_Y_RANGE = (-0.25, 0.25); CUBE_Z = 0.025
TARGET_X_RANGE  = (0.30, 0.55); TARGET_Y_RANGE = (-0.25, 0.25); TARGET_Z = 0.025
PLACE_Z_OFFSET  = 0.02
MIN_DISTANCE    = 0.15
WARMUP_STEPS    = 200

# 큐브 위치 격자 샘플링 설정
CUBE_GRID_NX       = 5     # x방향 격자점 수
CUBE_GRID_NY       = 5     # y방향 격자점 수  (5x5 = 25개 지점)
EPISODES_PER_POINT = 12    # 격자점당 수집 에피소드 수 (25 x 12 = 300 총 에피소드)
CUBE_JITTER        = 0.01  # 격자점 주변 소량 지터(±1cm). 완전히 똑같은 픽셀 프레임 방지용

START_POSE = [0.0, -0.3, 0.0, -2.5, 0.0, 2.2, 0.8, 0.04, 0.04]
START_SETTLE_STEPS = 60   
USE_START_POSE = True     
RECORD_RETREAT     = True
RETREAT_LIFT_STEPS = 50    
RETREAT_HOME_STEPS = 100   

RECORD_IMAGES = True
IMG_W, IMG_H = 160, 120                

SUCCESS_XY_TOL   = 0.05
SUCCESS_MIN_LIFT = 0.04

ENABLE_SMOOTHNESS_FILTER = True
MAX_JOINT_STEP_JUMP      = 0.30   
MAX_JOINT_VEL            = 25.0   
MAX_EE_DETOUR            = 1.8     

WRIST_CAM_PRIM = "/World/Franka/panda_hand/WristCam/Camera"
OVER_CAM_PRIM  = "/World/OverheadCam/Camera"
# =============================================================================

import os, sys, random
import numpy as np

os.makedirs(SAVE_PATH, exist_ok=True)

from isaacsim import SimulationApp
simulation_app = SimulationApp({"renderer": "RayTracedLighting", "headless": HEADLESS})

import omni.usd
from pxr import UsdGeom, Gf, UsdLux
from isaacsim.core.api import World
from isaacsim.core.api.objects import DynamicCuboid
from isaacsim.core.utils.types import ArticulationAction
from isaacsim.robot.manipulators.examples.franka import Franka
from isaacsim.robot.manipulators.examples.franka.controllers import PickPlaceController

def build_cube_schedule():
    """CUBE_GRID_NX x CUBE_GRID_NY 격자점 각각을 EPISODES_PER_POINT번씩 담은
    스케줄을 만들고 섞는다. 에피소드 인덱스 -> 격자점 (x,y) 매핑."""
    xs = np.linspace(CUBE_X_RANGE[0], CUBE_X_RANGE[1], CUBE_GRID_NX)
    ys = np.linspace(CUBE_Y_RANGE[0], CUBE_Y_RANGE[1], CUBE_GRID_NY)
    points = [(float(x), float(y)) for x in xs for y in ys]
    schedule = []
    for p in points:
        schedule += [p] * EPISODES_PER_POINT
    random.shuffle(schedule)
    return schedule

CUBE_SCHEDULE = build_cube_schedule()
NUM_EPISODES  = len(CUBE_SCHEDULE)

def sample_positions(ep):
    gx, gy = CUBE_SCHEDULE[ep]
    cube_x = gx + random.uniform(-CUBE_JITTER, CUBE_JITTER)
    cube_y = gy + random.uniform(-CUBE_JITTER, CUBE_JITTER)
    while True:
        tgt_x = random.uniform(*TARGET_X_RANGE)
        tgt_y = random.uniform(*TARGET_Y_RANGE)
        dist = np.sqrt((cube_x - tgt_x)**2 + (cube_y - tgt_y)**2)
        if dist >= MIN_DISTANCE:
            return np.array([cube_x, cube_y, CUBE_Z]), np.array([tgt_x, tgt_y, TARGET_Z])

def build_scene():
    world = World(stage_units_in_meters=1.0)
    world.scene.add_default_ground_plane()
    stage = omni.usd.get_context().get_stage()

    franka = world.scene.add(Franka(prim_path="/World/Franka", name="franka"))

    cube = world.scene.add(DynamicCuboid(
        prim_path="/World/PickCube", name="pick_cube",
        position=np.array([0.45, 0.0, CUBE_Z]), size=0.05,
        color=np.array([0.8, 0.2, 0.1]), mass=0.1
    ))

    marker_prim = stage.DefinePrim("/World/PlaceMarker", "Cylinder")
    UsdGeom.Cylinder(marker_prim).GetRadiusAttr().Set(0.025)
    UsdGeom.Cylinder(marker_prim).GetHeightAttr().Set(0.002)
    UsdGeom.Gprim(marker_prim).GetDisplayColorAttr().Set([Gf.Vec3f(0.1, 0.8, 0.1)])

    ee_path = "/World/Franka/panda_hand"
    wrist_xform = stage.DefinePrim(ee_path + "/WristCam", "Xform")
    wrist_cam = UsdGeom.Camera.Define(stage, WRIST_CAM_PRIM)
    wrist_cam.GetFocalLengthAttr().Set(16.0)   
    wrist_cam.GetHorizontalApertureAttr().Set(20.955)
    wrist_cam.GetVerticalApertureAttr().Set(15.716)
    wrist_cam.GetClippingRangeAttr().Set(Gf.Vec2f(0.05, 100.0))  
    UsdGeom.XformCommonAPI(wrist_xform).SetTranslate(Gf.Vec3d(0.15, 0.0, 0.0)) 
    UsdGeom.XformCommonAPI(wrist_xform).SetRotate(Gf.Vec3f(-45, 179.9, -89.9))

    over_xform = stage.DefinePrim("/World/OverheadCam", "Xform")
    over_cam = UsdGeom.Camera.Define(stage, OVER_CAM_PRIM)
    over_cam.GetFocalLengthAttr().Set(24.0)
    over_cam.GetHorizontalApertureAttr().Set(20.955)
    over_cam.GetVerticalApertureAttr().Set(15.716)
    over_cam.GetClippingRangeAttr().Set(Gf.Vec2f(0.05, 100.0))
    UsdGeom.XformCommonAPI(over_xform).SetTranslate(Gf.Vec3d(0.4, 0.0, 1.5))
    UsdGeom.XformCommonAPI(over_xform).SetRotate(Gf.Vec3f(0.0, 0.0, -89.9))

    for _ in range(50):
        simulation_app.update()

    import omni.replicator.core as rep
    rep.create.render_product(WRIST_CAM_PRIM, resolution=(IMG_W, IMG_H))
    rep.create.render_product(OVER_CAM_PRIM, resolution=(IMG_W, IMG_H))

    return world, franka, cube, marker_prim

def get_ee_position(franka):
    try:
        ee_pos, _ = franka.end_effector.get_world_pose()
        return np.array(ee_pos, dtype=np.float32)
    except Exception:
        stage = omni.usd.get_context().get_stage()
        hand = stage.GetPrimAtPath("/World/Franka/panda_hand")
        xf = UsdGeom.Xformable(hand).ComputeLocalToWorldTransform(0)
        t = xf.ExtractTranslation()
        return np.array([t[0], t[1], t[2]], dtype=np.float32)

def get_observation(franka, cube, target_pos):
    joint_pos = np.array(franka.get_joint_positions(),  dtype=np.float32)
    joint_vel = np.array(franka.get_joint_velocities(), dtype=np.float32)
    joint_vel = np.clip(joint_vel, -50.0, 50.0)
    cube_pos, _ = cube.get_world_pose()
    cube_pos = np.array(cube_pos, dtype=np.float32)
    ee_pos = get_ee_position(franka)
    cube_rel   = cube_pos - ee_pos               
    target_rel = target_pos.astype(np.float32) - ee_pos  
    return np.concatenate([joint_pos, joint_vel, cube_rel, target_rel, cube_pos])

def action_to_full(action, current_joint_pos, n_dof):
    full = current_joint_pos.copy()
    if action.joint_positions is None: return full
    jp = np.array(action.joint_positions, dtype=np.float32).flatten()
    if action.joint_indices is not None:
        indices = np.array(action.joint_indices, dtype=int).flatten()
        for i, idx in enumerate(indices):
            if 0 <= idx < n_dof and i < len(jp):
                if not np.isnan(jp[i]):
                    full[idx] = jp[i]
    elif len(jp) == n_dof:
        full = np.where(np.isnan(jp), full, jp)
    else:
        for i in range(min(len(jp), n_dof)):
            if not np.isnan(jp[i]):
                full[i] = jp[i]
    return full

def is_valid(obs, action):
    if np.isnan(obs).any() or np.isinf(obs).any(): return False
    if np.isnan(action).any() or np.isinf(action).any(): return False
    return True

world, franka, cube, marker_prim = build_scene()

_wrist_cam = _over_cam = None
_cams_inited = False

if RECORD_IMAGES:
    from isaacsim.sensors.camera import Camera
    _wrist_cam = Camera(prim_path=WRIST_CAM_PRIM, resolution=(IMG_W, IMG_H))
    _over_cam  = Camera(prim_path=OVER_CAM_PRIM,  resolution=(IMG_W, IMG_H))

def grab_rgb(cam):
    if cam is None: return np.zeros((IMG_H, IMG_W, 3), dtype=np.uint8)
    try:
        rgba = cam.get_rgba()
        if rgba is None or getattr(rgba, "ndim", 0) != 3 or rgba.shape[0] < 2:
            return np.zeros((IMG_H, IMG_W, 3), dtype=np.uint8)
        img = rgba[:, :, :3]
        if np.mean(img) > 250:
            simulation_app.update()
            rgba = cam.get_rgba()
            img = rgba[:, :, :3] if rgba is not None else img
        if img.dtype != np.uint8:
            img = (np.clip(img * 255.0, 0, 255) if float(img.max()) <= 1.0 else np.clip(img, 0, 255)).astype(np.uint8)
        return np.ascontiguousarray(img, dtype=np.uint8)
    except Exception as e:
        return np.zeros((IMG_H, IMG_W, 3), dtype=np.uint8)

stage = omni.usd.get_context().get_stage()
dome_light = UsdLux.DomeLight.Define(stage, "/World/DR_DomeLight")
cube_material = cube.get_applied_visual_material()

def randomize_domain():
    cube_material.set_color(np.array([random.uniform(0,1), random.uniform(0,1), random.uniform(0,1)]))
    dome_light.GetIntensityAttr().Set(random.uniform(500.0, 2500.0))

all_episodes  = []
skipped_total = n_dropped_erratic = n_dropped_fail_clean = 0
print(f"\n[collect] === V8 시작 (큐브 위치 격자 샘플링 {CUBE_GRID_NX}x{CUBE_GRID_NY}격자 x 지점당{EPISODES_PER_POINT}회 = {NUM_EPISODES} 에피소드) ===\n")

for ep in range(NUM_EPISODES):
    cube_pos, target_pos = sample_positions(ep)
    print(f"[collect] Episode {ep+1:3d}/{NUM_EPISODES} | cube={np.round(cube_pos,3)} | target={np.round(target_pos,3)}")

    world.reset()
    cube.set_world_pose(position=cube_pos)
    UsdGeom.XformCommonAPI(marker_prim).SetTranslate(Gf.Vec3d(*target_pos))

    if RECORD_IMAGES and not _cams_inited:
        _wrist_cam.initialize(); _over_cam.initialize()
        for _ in range(15): world.step(render=RENDER)
        _cams_inited = True

    randomize_domain()

    try: franka.gripper.open()
    except Exception: franka.gripper.apply_action(ArticulationAction(joint_positions=franka.gripper.joint_opened_positions))
    for _ in range(20): world.step(render=RENDER)

    for _ in range(WARMUP_STEPS): world.step(render=RENDER)

    if USE_START_POSE:
        n_dof_tmp = franka.num_dof
        start_full = np.array(START_POSE, dtype=float)
        if len(start_full) < n_dof_tmp:
            cur = np.array(franka.get_joint_positions(), dtype=float)
            cur[:len(start_full)] = start_full; start_full = cur
        for _ in range(START_SETTLE_STEPS):
            franka.apply_action(ArticulationAction(joint_positions=start_full))
            world.step(render=RENDER)

    n_dof = franka.num_dof
    controller = PickPlaceController(name="pick_place_controller", gripper=franka.gripper, robot_articulation=franka)

    ep_obs, ep_actions, ep_wrist, ep_over = [], [], [], []
    skipped_ep = max_cube_z = max_joint_jump = max_joint_vel = ee_path_xy = 0
    prev_arm_q = prev_ee_xy = ee_start_xy = None
    controller_done = False; step = 0

    while step < MAX_STEPS:
        current_joint_pos = np.array(franka.get_joint_positions(), dtype=np.float32)
        obs = get_observation(franka, cube, target_pos)

        arm_q = current_joint_pos[:7]
        if prev_arm_q is not None: max_joint_jump = max(max_joint_jump, float(np.max(np.abs(arm_q - prev_arm_q))))
        prev_arm_q = arm_q
        raw_vel = np.array(franka.get_joint_velocities(), dtype=np.float32)
        max_joint_vel = max(max_joint_vel, float(np.max(np.abs(raw_vel[:7]))))

        ee_xy = get_ee_position(franka)[:2]
        if ee_start_xy is None: ee_start_xy = ee_xy
        if prev_ee_xy is not None: ee_path_xy += float(np.linalg.norm(ee_xy - prev_ee_xy))
        prev_ee_xy = ee_xy

        place_target = target_pos.copy(); place_target[2] = target_pos[2] + PLACE_Z_OFFSET

        # 1. 시뮬레이터 구동용 액션 계산 (원본 그대로 사용하여 스무스하게 이동)
        action = controller.forward(picking_position=cube_pos, placing_position=place_target, current_joint_positions=current_joint_pos)
        clean_action = action_to_full(action, current_joint_pos, n_dof)
        
        # 2. 데이터셋 저장용 액션 -> 그리퍼 이진화 (안전하게 0.025 / 0.04)
        stored_action = clean_action.copy()
        is_closing = stored_action[7] < 0.035
        stored_action[7] = 0.025 if is_closing else 0.04
        stored_action[8] = 0.025 if is_closing else 0.04

        if is_valid(obs, stored_action):
            ep_obs.append(obs)
            ep_actions.append(stored_action)  # 노이즈가 제거된 정답을 저장
            if RECORD_IMAGES:
                ep_wrist.append(grab_rgb(_wrist_cam))
                ep_over.append(grab_rgb(_over_cam))
        else: skipped_ep += 1

        # 원본 액션 그대로 시뮬레이터에 적용 (물리 엔진 평화 유지)
        franka.apply_action(action)
        world.step(render=RENDER)
        step += 1

        cube_world_pos, _ = cube.get_world_pose()
        max_cube_z = max(max_cube_z, float(cube_world_pos[2]))

        if controller.is_done():
            controller_done = True
            break

    skipped_total += skipped_ep

    if RECORD_RETREAT and controller_done:
        home_q = np.array(START_POSE, dtype=np.float32)
        if len(home_q) < n_dof:
            pad = np.array(franka.get_joint_positions(), dtype=np.float32)
            pad[:len(home_q)] = home_q; home_q = pad
        cur_q = np.array(franka.get_joint_positions(), dtype=np.float32)
        lift_q = cur_q.copy()
        lift_q[1] = home_q[1]; lift_q[3] = home_q[3]; lift_q[7:9] = 0.04
        for target_q, n_steps in [(lift_q, RETREAT_LIFT_STEPS), (home_q, RETREAT_HOME_STEPS)]:
            q0 = np.array(franka.get_joint_positions(), dtype=np.float32)
            for s in range(n_steps):
                alpha = float(s + 1) / n_steps
                clean_cmd = ((1.0 - alpha) * q0 + alpha * target_q).astype(np.float32)
                clean_cmd[7:9] = 0.04  
                
                obs = get_observation(franka, cube, target_pos)
                stored_action = clean_cmd.copy()
                
                if is_valid(obs, stored_action):
                    ep_obs.append(obs)
                    ep_actions.append(stored_action)
                    if RECORD_IMAGES:
                        ep_wrist.append(grab_rgb(_wrist_cam))
                        ep_over.append(grab_rgb(_over_cam))
                franka.apply_action(ArticulationAction(joint_positions=clean_cmd))
                world.step(render=RENDER)

    final_cube_pos, _ = cube.get_world_pose()
    final_xy_err = float(np.linalg.norm(np.array(final_cube_pos, dtype=np.float32)[:2] - target_pos[:2]))
    lifted = (max_cube_z - CUBE_Z) >= SUCCESS_MIN_LIFT
    placed = final_xy_err <= SUCCESS_XY_TOL
    success = lifted and placed

    ideal_xy = (np.linalg.norm(ee_start_xy - cube_pos[:2]) + np.linalg.norm(cube_pos[:2] - target_pos[:2])) if ee_start_xy is not None else 0.0
    ee_detour = ee_path_xy / max(float(ideal_xy), 0.05)

    jumpy = (max_joint_jump > MAX_JOINT_STEP_JUMP) or (max_joint_vel > MAX_JOINT_VEL)
    wander = (ee_detour > MAX_EE_DETOUR)
    erratic = jumpy or wander

    if ENABLE_SMOOTHNESS_FILTER and erratic: n_dropped_erratic += 1; continue
    if not success: n_dropped_fail_clean += 1; continue
    if len(ep_obs) == 0: continue

    ep_data = {
        "obs": np.array(ep_obs, dtype=np.float32),
        "actions": np.array(ep_actions, dtype=np.float32),
        "cube_pos": cube_pos, "target_pos": target_pos,
    }
    all_episodes.append(ep_data)

    save_data = dict(ep_data)
    if RECORD_IMAGES:
        save_data["images_wrist"] = np.array(ep_wrist, dtype=np.uint8)
        save_data["images_over"]  = np.array(ep_over,  dtype=np.uint8)
    ep_save_path = os.path.join(SAVE_PATH, f"episode_{ep:04d}.npz")
    np.savez_compressed(ep_save_path, **save_data)

if len(all_episodes) == 0:
    print("\n[collect] 치명적 오류: 성공한 에피소드가 없습니다.")
    import omni.replicator.core as rep
    rep.orchestrator.stop()
    simulation_app.close()
    sys.exit(1)

all_obs = np.concatenate([ep["obs"] for ep in all_episodes], axis=0)
all_actions = np.concatenate([ep["actions"] for ep in all_episodes], axis=0)
ep_lengths = [len(ep["obs"]) for ep in all_episodes]
episode_starts = np.cumsum([0] + ep_lengths[:-1]).astype(np.int64)

merged_path = os.path.join(SAVE_PATH, "bc_dataset.npz")
np.savez(merged_path, obs=all_obs, actions=all_actions, episode_starts=episode_starts,
         start_pose=np.array(START_POSE, dtype=np.float32), use_delta=np.array([0], dtype=np.int32))

print(f"[collect] 저장 완료! (에피소드 {len(all_episodes)}개)")
import omni.replicator.core as rep
rep.orchestrator.stop()
simulation_app.close()