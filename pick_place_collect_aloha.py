# pick_place_collect_aloha.py  (Isaac Sim 5.1)
# ---------------------------------------------------------------------------
# BC 데이터 수집 v9 - ALOHA 표준 세팅: "큐브 랜덤 + 목표 고정" 50 에피소드
#
# 왜 이렇게 바꿨나
# ---------------
# v8은 큐브를 5x5 격자에 12번씩 = 300 에피소드를 모았지만, 학습 결과가
# 격자점(in-distribution)에서조차 성공률 0%였다. 원인은 두 가지였다.
#
#  (1) 학습량 부족: lerobot 기본값(100k step x batch 8 = 80만 샘플)을 31만
#      프레임에 쓰면 2.6 에폭밖에 안 된다. ACT 표준 레시피는 수십 에폭이다.
#  (2) **목표 위치가 통제된 적이 없다**: v8은 큐브만 격자로 고정하고 놓을
#      지점은 TARGET_*_RANGE 전체에서 매번 새로 뽑았다. 즉 300 에피소드가
#      전부 서로 다른 목표를 가진 셈이다.
#
# ACT/ALOHA 계열에서 "50 에피소드면 된다"는 결과들은 전부 다음 세팅이다:
#   - 물체(큐브) 초기 위치: **랜덤**  (그래서 일반화가 된다)
#   - 목표(놓을 곳):        **고정**  (ALOHA sim_transfer_cube = fixed bin)
# v8은 이 중 두 번째를 지키지 않았다. v9는 그 표준을 그대로 따른다.
#
# 50 x 1067 = 53,350 프레임. 3프레임마다 솎으면 17,800 프레임이라
# lerobot 기본값 100k x 8이 45 에폭이 된다 (v5는 2.6이었다).
#
# 성공률이 올라온 뒤에 목표 지점도 랜덤으로 넓히는 게 다음 단계다. 순서를
# 뒤집지 말 것 - 변량을 하나씩 풀어야 뭐가 원인인지 알 수 있다.
#
# 실행(워크스테이션):  /data/isaacsim/python.sh pick_place_collect_aloha.py
# === 설정 ===================================================================
# 카메라/해상도/조명/영역 값은 scene_config.py 한 곳에만 둔다. 예전엔 이 파일과
# eval_act_v5_client.py에 각각 복사돼 있었고 실제로 어긋나서(조명 1500 vs 1000)
# 정책이 학습한 적 없는 이미지로 평가됐다.
from scene_config import (
    IMG_W, IMG_H, WRIST_CAM_PRIM, OVER_CAM_PRIM,
    CAM_H_APERTURE, CAM_V_APERTURE, CAM_CLIP_NEAR, CAM_CLIP_FAR,
    WRIST_FOCAL, WRIST_TRANSLATE, WRIST_ROTATE,
    OVER_FOCAL, OVER_TRANSLATE, OVER_ROTATE,
    LIGHT_INTENSITY, CUBE_COLOR, CUBE_SIZE, CUBE_MASS, CUBE_Z, TARGET_Z,
    TARGET_FIXED_XY, TARGET_JITTER, CUBE_X_RANGE, CUBE_Y_RANGE, MIN_DISTANCE,
    START_POSE, SUCCESS_XY_TOL, SUCCESS_MIN_LIFT, NUM_EPISODES,
    GRIPPER_CLOSED, GRIPPER_OPEN, GRIPPER_CLOSING_RAW_THRESH,
)

MAX_STEPS       = 2000
HEADLESS        = True
RENDER          = True
SAVE_PATH       = r"/data/jinju/bc_data_v11"

PLACE_Z_OFFSET  = 0.02
WARMUP_STEPS    = 200

# 도메인 랜덤화. v8은 매 에피소드 큐브 색을 완전 랜덤 RGB로 바꿨는데, ALOHA
# 표준 세팅에는 그런 게 없다. 50개로 색까지 배우게 하면 부담이 크므로 끈다.
# 성공률이 올라온 뒤에 다시 켜서 강건성을 본다.
DOMAIN_RANDOMIZE = False

START_SETTLE_STEPS = 60
USE_START_POSE = True
RECORD_RETREAT     = True
RETREAT_LIFT_STEPS = 50
RETREAT_HOME_STEPS = 100

RECORD_IMAGES = True

ENABLE_SMOOTHNESS_FILTER = True
MAX_JOINT_STEP_JUMP      = 0.30
MAX_JOINT_VEL            = 25.0
MAX_EE_DETOUR            = 1.8
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

# ---------------------------------------------------------------------------
# 이어받기(RESUME): SAVE_PATH에 이미 episode_*.npz가 있으면 그 개수만큼 빼고
# 부족분만 더 모은다. v8은 격자점별로 집계해야 했지만 v9는 지점이 하나라
# 파일 개수만 세면 된다. (중간에 멈춰도 처음부터 다시 돌리지 않기 위함)
#
# 단, 랜덤 범위(CUBE_*_RANGE)를 바꾼 뒤에 이어받으면 서로 다른 분포의 에피소드가
# 한 데이터셋에 섞인다. 그럴 때는 FRESH_START=1로 기존 파일을 지우고 새로 모을 것:
#     FRESH_START=1 /data/isaacsim/python.sh pick_place_collect_aloha.py
# 기본값은 0(이어받기)이다 - 실수로 수집분을 날리지 않기 위함.
# ---------------------------------------------------------------------------
FRESH_START = os.environ.get("FRESH_START", "0") == "1"


def build_resume_plan():
    """반환: (이번 회차에 모을 에피소드 수, 저장 시작 인덱스)."""
    import glob as _glob
    files = sorted(_glob.glob(os.path.join(SAVE_PATH, "episode_*.npz")))

    if FRESH_START:
        stale = files + _glob.glob(os.path.join(SAVE_PATH, "bc_dataset.npz"))
        for f in stale:
            os.remove(f)
        print(f"[fresh] FRESH_START=1 → 기존 파일 {len(stale)}개 삭제, "
              f"{NUM_EPISODES}개를 처음부터 모은다")
        return NUM_EPISODES, 0

    if not files:
        return NUM_EPISODES, 0
    max_idx = -1
    for f in files:
        try:
            max_idx = max(max_idx, int(os.path.basename(f).split("_")[1].split(".")[0]))
        except Exception:
            pass
    todo = max(0, NUM_EPISODES - len(files))
    print(f"[resume] 기존 파일 {len(files)}개 발견 (목표 {NUM_EPISODES}개) => "
          f"추가 수집 {todo}개 | 저장 인덱스 {max_idx + 1}부터 (기존 파일 보존)")
    return todo, max_idx + 1

N_TO_COLLECT, SAVE_INDEX_OFFSET = build_resume_plan()

def sample_positions(ep):
    """큐브는 연속 랜덤, 목표는 고정(+미세 지터). (ep는 안 쓰지만 시그니처 유지)"""
    tgt_x = TARGET_FIXED_XY[0] + random.uniform(-TARGET_JITTER, TARGET_JITTER)
    tgt_y = TARGET_FIXED_XY[1] + random.uniform(-TARGET_JITTER, TARGET_JITTER)
    while True:
        cube_x = random.uniform(*CUBE_X_RANGE)
        cube_y = random.uniform(*CUBE_Y_RANGE)
        if np.hypot(cube_x - tgt_x, cube_y - tgt_y) >= MIN_DISTANCE:
            return (np.array([cube_x, cube_y, CUBE_Z]),
                    np.array([tgt_x,  tgt_y,  TARGET_Z]))

def build_scene():
    world = World(stage_units_in_meters=1.0)
    world.scene.add_default_ground_plane()
    stage = omni.usd.get_context().get_stage()

    franka = world.scene.add(Franka(prim_path="/World/Franka", name="franka"))

    cube = world.scene.add(DynamicCuboid(
        prim_path="/World/PickCube", name="pick_cube",
        position=np.array([0.45, 0.0, CUBE_Z]), size=CUBE_SIZE,
        color=np.array(CUBE_COLOR), mass=CUBE_MASS
    ))

    marker_prim = stage.DefinePrim("/World/PlaceMarker", "Cylinder")
    UsdGeom.Cylinder(marker_prim).GetRadiusAttr().Set(0.025)
    UsdGeom.Cylinder(marker_prim).GetHeightAttr().Set(0.002)
    UsdGeom.Gprim(marker_prim).GetDisplayColorAttr().Set([Gf.Vec3f(0.1, 0.8, 0.1)])

    ee_path = "/World/Franka/panda_hand"
    wrist_xform = stage.DefinePrim(ee_path + "/WristCam", "Xform")
    wrist_cam = UsdGeom.Camera.Define(stage, WRIST_CAM_PRIM)
    wrist_cam.GetFocalLengthAttr().Set(WRIST_FOCAL)
    wrist_cam.GetHorizontalApertureAttr().Set(CAM_H_APERTURE)
    wrist_cam.GetVerticalApertureAttr().Set(CAM_V_APERTURE)
    wrist_cam.GetClippingRangeAttr().Set(Gf.Vec2f(CAM_CLIP_NEAR, CAM_CLIP_FAR))
    UsdGeom.XformCommonAPI(wrist_xform).SetTranslate(Gf.Vec3d(*WRIST_TRANSLATE))
    UsdGeom.XformCommonAPI(wrist_xform).SetRotate(Gf.Vec3f(*WRIST_ROTATE))

    over_xform = stage.DefinePrim("/World/OverheadCam", "Xform")
    over_cam = UsdGeom.Camera.Define(stage, OVER_CAM_PRIM)
    over_cam.GetFocalLengthAttr().Set(OVER_FOCAL)
    over_cam.GetHorizontalApertureAttr().Set(CAM_H_APERTURE)
    over_cam.GetVerticalApertureAttr().Set(CAM_V_APERTURE)
    over_cam.GetClippingRangeAttr().Set(Gf.Vec2f(CAM_CLIP_NEAR, CAM_CLIP_FAR))
    UsdGeom.XformCommonAPI(over_xform).SetTranslate(Gf.Vec3d(*OVER_TRANSLATE))
    UsdGeom.XformCommonAPI(over_xform).SetRotate(Gf.Vec3f(*OVER_ROTATE))

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
    """DOMAIN_RANDOMIZE=False면 색·조명을 매 에피소드 같은 값으로 고정한다.
    (끄는 게 아니라 고정값을 다시 써준다 - 이전 에피소드 값이 남지 않도록)"""
    if DOMAIN_RANDOMIZE:
        cube_material.set_color(np.array([random.uniform(0,1), random.uniform(0,1), random.uniform(0,1)]))
        dome_light.GetIntensityAttr().Set(random.uniform(500.0, 2500.0))
    else:
        cube_material.set_color(np.array(CUBE_COLOR))
        dome_light.GetIntensityAttr().Set(LIGHT_INTENSITY)

all_episodes  = []
skipped_total = n_dropped_erratic = n_dropped_fail_clean = 0
print(f"\n[collect] === V9 시작 (ALOHA 표준: 큐브 랜덤 + 목표 고정) ===")
print(f"[collect]   큐브   랜덤  x{CUBE_X_RANGE} y{CUBE_Y_RANGE}  "
      f"({(CUBE_X_RANGE[1]-CUBE_X_RANGE[0])*100:.0f}x{(CUBE_Y_RANGE[1]-CUBE_Y_RANGE[0])*100:.0f}cm)")
print(f"[collect]   타겟   고정  {TARGET_FIXED_XY}  (지터 ±{TARGET_JITTER*1000:.0f}mm)")
print(f"[collect]   목표   {NUM_EPISODES} 에피소드 | 이번 회차 {N_TO_COLLECT}개")
print(f"[collect]   도메인 랜덤화 {'ON' if DOMAIN_RANDOMIZE else 'OFF (색·조명 고정)'}")
print(f"[collect]   저장   {SAVE_PATH}\n")

if N_TO_COLLECT == 0:
    print("[collect] 이미 목표 개수를 채웠습니다. 더 모으려면 NUM_EPISODES를 올리세요.")
    import omni.replicator.core as rep
    rep.orchestrator.stop()
    simulation_app.close()
    sys.exit(0)

for ep in range(N_TO_COLLECT):
    cube_pos, target_pos = sample_positions(ep)
    print(f"[collect] Episode {ep+1:3d}/{N_TO_COLLECT} | cube={np.round(cube_pos,3)} | target={np.round(target_pos,3)}")

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
    # 컨트롤러가 '닫을 때' 로봇에 실제로 주는 원본 값. 라벨(GRIPPER_CLOSED)이
    # 이 값과 크게 다르면 v10과 같은 실패가 재발하므로 매 에피소드 찍어서 감시한다.
    raw_closing_cmds = []
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
        
        # 2. 데이터셋 저장용 액션 -> 그리퍼 이진화
        # [!] 여기 값이 로봇에 실제로 가는 값(아래 apply_action(action))과 같아야 한다.
        #   v10까지는 라벨을 0.025로 저장했는데, 0.025는 손가락당 2.5cm =
        #   총 개구 5.0cm로 큐브 폭과 정확히 같아서 미는 힘이 0이다. 정책이 배운
        #   대로 명령하면 닿기만 하고 못 쥐어서 성공률이 0%였다. 컨트롤러가 로봇에
        #   주는 닫힘 명령은 그보다 작은 값이므로, 라벨도 GRIPPER_CLOSED(0.0)로 둔다.
        stored_action = clean_action.copy()
        is_closing = stored_action[7] < GRIPPER_CLOSING_RAW_THRESH
        stored_action[7] = GRIPPER_CLOSED if is_closing else GRIPPER_OPEN
        stored_action[8] = GRIPPER_CLOSED if is_closing else GRIPPER_OPEN
        if is_closing:
            raw_closing_cmds.append(float(clean_action[7]))

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
    ep_save_path = os.path.join(SAVE_PATH, f"episode_{ep + SAVE_INDEX_OFFSET:04d}.npz")
    np.savez_compressed(ep_save_path, **save_data)

    # 라벨과 실제 명령이 일치하는지 감시 (v10을 0%로 만든 원인이 이 불일치였다)
    if raw_closing_cmds:
        lo, hi = min(raw_closing_cmds), max(raw_closing_cmds)
        note = "" if abs(hi - GRIPPER_CLOSED) < 0.01 else \
            f"  [!] 라벨({GRIPPER_CLOSED})과 차이가 크다 - 이 값으로 라벨을 맞출 것"
        print(f"[collect]   저장 {len(ep_obs)}프레임 | 닫는 명령 원본 "
              f"{lo:.4f}~{hi:.4f} (라벨 {GRIPPER_CLOSED}){note}")

if len(all_episodes) == 0:
    if SAVE_INDEX_OFFSET > 0:
        print("\n[resume] 새로 수집한 에피소드가 없습니다 "
              "(모든 격자점이 이미 목표치를 채웠거나 이번 회차가 전부 드롭됨).")
        import omni.replicator.core as rep
        rep.orchestrator.stop()
        simulation_app.close()
        sys.exit(0)
    print("\n[collect] 치명적 오류: 성공한 에피소드가 없습니다.")
    import omni.replicator.core as rep
    rep.orchestrator.stop()
    simulation_app.close()
    sys.exit(1)

# 병합 파일(bc_dataset.npz)은 변환 파이프라인에서 사용하지 않고(convert는 episode_*.npz만 읽음),
# 이어받기 시에는 이번 회차분만 담겨 기존 병합본을 덮어쓰게 되므로 '새 수집'일 때만 기록한다.
if SAVE_INDEX_OFFSET == 0:
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