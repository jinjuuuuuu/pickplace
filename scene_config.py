#!/usr/bin/env python3
# scene_config.py - 수집과 평가가 공유하는 씬 설정 (숫자만 담는다)
# ---------------------------------------------------------------------------
# 왜 이 파일이 있나
# ---------------
# 카메라 위치/focal/해상도/조명 값이 pick_place_collect_aloha.py 와
# eval_act_v5_client.py 에 각각 복사돼 있었고, 실제로 어긋났다:
#   - 조명: 수집 1500 vs 평가 1000  (정책이 학습한 적 없는 밝기로 평가됨)
# 수집과 평가의 씬이 다르면 정책은 학습 분포 밖의 이미지를 받고, 아무리 잘
# 학습해도 0%가 나온다. 그 종류의 버그를 구조적으로 막기 위해 값을 여기 모았다.
#
# 이 파일은 순수 상수 모듈이다 (isaacsim/pxr import 없음) -> conda 환경에서도,
# Isaac Sim python에서도, 아무 곳에서나 import 된다.
#
# [!] 카메라나 해상도를 바꾸면 반드시 데이터를 다시 수집해야 한다. 이미지가
#   npz 안에 구워져 있으므로, 기존 데이터와 새 설정을 섞으면 안 된다.
# ---------------------------------------------------------------------------

import os as _os

# === 이미지 =================================================================
# 160x120 -> 320x240. 기존엔 5cm 큐브가 오버헤드에서 6px밖에 안 되어 ACT가
# 위치를 회귀할 수 없었다. (측정: diag_dataset_images.py)
IMG_W, IMG_H = 320, 240

# === 카메라 프림 경로 =======================================================
WRIST_CAM_PRIM = "/World/Franka/panda_hand/WristCam/Camera"
OVER_CAM_PRIM = "/World/OverheadCam/Camera"

# 두 카메라 공통 (Isaac Sim 기본 센서 크기)
CAM_H_APERTURE = 20.955
CAM_V_APERTURE = 15.716
CAM_CLIP_NEAR, CAM_CLIP_FAR = 0.05, 100.0

# === 손목 카메라 (변경 없음) ================================================
# 큐브를 18~50px로 보고 있어 유일하게 제대로 작동했던 센서다. 단, 팔이 시작
# 자세일 때 큐브가 x>=0.5 에 있으면 화각을 벗어난다(측정: cam_tune.py) -
# 그 구간은 오버헤드가 담당한다.
WRIST_FOCAL = 16.0
WRIST_TRANSLATE = (0.15, 0.0, 0.0)
WRIST_ROTATE = (-45.0, 179.9, -89.9)

# === 오버헤드 카메라 (전면 교체) ============================================
# 기존: 천장 수직뷰 (0.4, 0, 1.5) rot(0,0,-89.9) focal 24
#   -> 팔이 큐브를 가려서 에피소드의 대부분 구간에서 큐브가 0px였다.
#      (측정: diag_dataset_images.py - t=0에 5px, t=60 이후 계속 0px)
# 변경: 정면에서 50도 내려보는 경사뷰. 팔이 큐브 뒤쪽에서 접근해 가림이 적다.
#   focal 60mm 근거: 작업영역까지 2.55m라 기존 화각 유지에만 41mm가 필요하다.
#   60mm면 큐브 18~20 x 23~26px (다섯 위치 실측), 커버리지 0.87 x 0.85m로
#   로봇 베이스부터 목표 마커까지 프레임에 들어온다. 97mm를 넘으면 목표가 잘린다.
OVER_FOCAL = 60.0
OVER_TRANSLATE = (2.0, 0.0, 2.0)
OVER_ROTATE = (40.0, 0.0, 89.99)

# === 조명 / 물체 ============================================================
LIGHT_INTENSITY = 1500.0        # 수집·평가가 같아야 한다 (한 번 어긋나서 0% 났음)
CUBE_COLOR = (0.8, 0.2, 0.1)
CUBE_SIZE = 0.05
CUBE_MASS = 0.1
CUBE_Z = 0.025
TARGET_Z = 0.025

# === 태스크 배치 ============================================================
# ALOHA 표준: 물체는 랜덤, 놓을 곳은 고정.
TARGET_FIXED_XY = (0.500, -0.150)
TARGET_JITTER = 0.005           # ±5mm. 성공 허용오차(5cm)의 1/10.

# 큐브 랜덤 **수집** 영역 = 24x24cm, 중심 (0.425, 0.150).
# y를 전부 양수로 두는 이유: 목표가 (0.500,-0.150)이라 MIN_DISTANCE 조건을
# 영역 '전체'가 만족해야 사각형이 깎이지 않는다. 이 영역의 목표까지 최소거리는
# 0.186m. 대신 y<0 쪽에서 집는 건 배우지 못한다.
#
# v10은 이 영역이 20x20cm였고 평가 격자가 그 경계에 딱 걸쳐 있었다. 결과:
# 경계행(y=0.050)이 1/4, 내부는 11/12 (전체 75%). 연속 랜덤 수집에서 경계점은
# 이웃 데이터가 한쪽에만 있어서 학습이 얇아진다. 그래서 수집 영역을 평가
# 격자보다 EVAL_MARGIN만큼 넓게 잡는다 -> 평가 위치가 전부 내부점이 된다.
CUBE_X_RANGE = (0.305, 0.545)
CUBE_Y_RANGE = (0.03, 0.27)
MIN_DISTANCE = 0.15             # 큐브가 목표에 붙으면 태스크가 성립 안 함

# 수집 규모. 영역이 400 -> 576cm^2로 넓어졌으므로 v10과 같은 위치당 밀도를
# 유지하려면 47 * 576/400 = 68개가 필요하다. 70개로 잡는다.
NUM_EPISODES = 70

START_POSE = [0.0, -0.3, 0.0, -2.5, 0.0, 2.2, 0.8, 0.04, 0.04]

# === 그리퍼 =================================================================
# [!] v10에서 0%를 만든 마지막 원인. 수집 스크립트는 로봇에 컨트롤러 원본 명령을
#   주면서(franka.apply_action(action)) 데이터셋에는 0.025로 덮어써 저장했다.
#   0.025는 손가락당 2.5cm = 총 개구 5.0cm로 큐브 폭(5cm)과 정확히 같아서,
#   정책이 배운 대로 0.025를 명령하면 닿기만 하고 미는 힘이 0이다 -> 못 쥔다.
#   같은 체크포인트에서 닫힘 명령만 0.0으로 바꾸자 0% -> 75%가 됐다.
# 그래서 라벨과 실제 명령을 모두 이 두 값으로 통일한다.
GRIPPER_CLOSED = 0.0            # 완전히 닫으라고 명령 -> 큐브가 막아 0.025에서 멈추며 쥔다
GRIPPER_OPEN = 0.04
# 정책 출력(회귀값)을 이진화할 때 닫힘으로 볼 임계값
GRIPPER_CLOSE_THRESH = (GRIPPER_CLOSED + GRIPPER_OPEN) / 2.0    # 0.02
# 수집 시 원본 액션이 '닫는 중'인지 판정하는 값 (컨트롤러 원본값 기준)
GRIPPER_CLOSING_RAW_THRESH = 0.035

# === 평가 위치 ==============================================================
# 수집 영역에서 EVAL_MARGIN만큼 안쪽으로 들어온 NxN 격자. 범위에서 자동
# 계산하므로 CUBE_*_RANGE를 바꾸면 평가 위치도 따라온다 (예전엔 손으로 맞춰야
# 해서 평가 위치의 절반이 학습 영역 밖이었다 = 무조건 실패).
# 학습은 연속 랜덤이므로 이 격자점들은 전부 '처음 보는 정확한 좌표'다.
# EVAL_MARGIN=0.02면 격자가 v10 평가와 똑같은 16개 좌표가 되어 직접 비교된다.
EVAL_GRID_N = int(_os.environ.get("EVAL_GRID_N", "4"))
# 양수면 수집 영역 안쪽, **음수면 바깥쪽** 격자가 된다. 음수로 주면 학습 분포
# 밖에서의 일반화를 시험할 수 있다 (기본 평가는 분포 내 보간일 뿐이다).
#   EVAL_MARGIN=-0.04 -> 수집 영역보다 4cm 넓은 격자 = 학습한 적 없는 영역
EVAL_MARGIN = float(_os.environ.get("EVAL_MARGIN", "0.02"))


def _lin(a, b, n):
    if n == 1:
        return [(a + b) / 2.0]
    return [a + (b - a) * i / (n - 1) for i in range(n)]


EVAL_CUBE_XY_LIST = [
    (round(x, 3), round(y, 3))
    for x in _lin(CUBE_X_RANGE[0] + EVAL_MARGIN, CUBE_X_RANGE[1] - EVAL_MARGIN, EVAL_GRID_N)
    for y in _lin(CUBE_Y_RANGE[0] + EVAL_MARGIN, CUBE_Y_RANGE[1] - EVAL_MARGIN, EVAL_GRID_N)
]

# === 기록 주기 / 솎기 ========================================================
# Isaac Sim World 기본 physics_dt=1/60이고 world.step()마다 한 프레임 저장한다.
# 이건 참고 사례들보다 촘촘하다 (ALOHA 50Hz/400프레임, Deepkar 30Hz/450프레임).
# 우리는 1067프레임/에피소드이므로 솎아서 저쪽 주기에 맞춘다.
#
# TRAIN_STRIDE를 바꾸면 세 곳이 같이 따라와야 한다. 그래서 여기 한 곳에 둔다:
#   1) subsample_dataset.py --stride 의 기본값
#   2) 평가의 ACTION_REPEAT (정책은 RECORD_HZ/STRIDE Hz로 판단, 물리는 60Hz)
#   3) chunk 지평 (chunk_size 100이 에피소드의 몇 %를 덮는가)
# stride별 지평:  1 -> 9.4%   2 -> 18.7%   3 -> 28.1%
#   비교: ALOHA 25%, Deepkar 22%. -> 3이 표준 영역이다.
RECORD_HZ = 60
TRAIN_STRIDE = 3

# 평가 성공 판정
SUCCESS_XY_TOL = 0.05
SUCCESS_MIN_LIFT = 0.04


if __name__ == "__main__":
    print(f"이미지        {IMG_W}x{IMG_H}")
    print(f"오버헤드      pos={OVER_TRANSLATE} rot={OVER_ROTATE} focal={OVER_FOCAL}")
    print(f"손목          pos={WRIST_TRANSLATE} rot={WRIST_ROTATE} focal={WRIST_FOCAL}")
    print(f"조명          {LIGHT_INTENSITY}")
    print(f"수집 영역     x{CUBE_X_RANGE} y{CUBE_Y_RANGE} "
          f"({(CUBE_X_RANGE[1]-CUBE_X_RANGE[0])*100:.0f}x"
          f"{(CUBE_Y_RANGE[1]-CUBE_Y_RANGE[0])*100:.0f}cm) x {NUM_EPISODES}회")
    print(f"목표          {TARGET_FIXED_XY} ±{TARGET_JITTER*1000:.0f}mm")
    print(f"그리퍼        닫힘 {GRIPPER_CLOSED} / 열림 {GRIPPER_OPEN} "
          f"(이진화 임계 {GRIPPER_CLOSE_THRESH})")
    print(f"솎기          stride {TRAIN_STRIDE} ({RECORD_HZ}Hz -> "
          f"{RECORD_HZ//TRAIN_STRIDE}Hz)")
    _where = (f"수집 영역에서 {EVAL_MARGIN*100:.0f}cm 안쪽" if EVAL_MARGIN >= 0
              else f"수집 영역보다 {-EVAL_MARGIN*100:.0f}cm 밖 [!] 학습 분포 밖")
    print(f"평가 위치     {len(EVAL_CUBE_XY_LIST)}개 ({EVAL_GRID_N}x{EVAL_GRID_N} 격자, {_where})")
    for i in range(0, len(EVAL_CUBE_XY_LIST), EVAL_GRID_N):
        print("              " + "  ".join(
            f"({x:.3f},{y:.3f})" for x, y in EVAL_CUBE_XY_LIST[i:i + EVAL_GRID_N]))
