#!/usr/bin/env python3
# scene_config.py — 수집과 평가가 공유하는 씬 설정 (숫자만 담는다)
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
# ⚠ 카메라나 해상도를 바꾸면 반드시 데이터를 다시 수집해야 한다. 이미지가
#   npz 안에 구워져 있으므로, 기존 데이터와 새 설정을 섞으면 안 된다.
# ---------------------------------------------------------------------------

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
# 자세일 때 큐브가 x>=0.5 에 있으면 화각을 벗어난다(측정: cam_tune.py) —
# 그 구간은 오버헤드가 담당한다.
WRIST_FOCAL = 16.0
WRIST_TRANSLATE = (0.15, 0.0, 0.0)
WRIST_ROTATE = (-45.0, 179.9, -89.9)

# === 오버헤드 카메라 (전면 교체) ============================================
# 기존: 천장 수직뷰 (0.4, 0, 1.5) rot(0,0,-89.9) focal 24
#   -> 팔이 큐브를 가려서 에피소드의 대부분 구간에서 큐브가 0px였다.
#      (측정: diag_dataset_images.py — t=0에 5px, t=60 이후 계속 0px)
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

# 큐브 랜덤 영역 = 정확히 20x20cm, 중심 (0.425, 0.150).
# y를 전부 양수로 두는 이유: 목표가 (0.500,-0.150)이라 MIN_DISTANCE 조건을
# 영역 '전체'가 만족해야 사각형이 깎이지 않는다. 이 영역의 목표까지 최소거리는
# 0.202m. 대신 y<0 쪽에서 집는 건 배우지 못한다.
CUBE_X_RANGE = (0.325, 0.525)
CUBE_Y_RANGE = (0.05, 0.25)
MIN_DISTANCE = 0.15             # 큐브가 목표에 붙으면 태스크가 성립 안 함

START_POSE = [0.0, -0.3, 0.0, -2.5, 0.0, 2.2, 0.8, 0.04, 0.04]

# === 평가 위치 ==============================================================
# 학습 영역을 고르게 덮는 NxN 격자. 범위에서 자동으로 계산하므로 CUBE_*_RANGE를
# 바꾸면 평가 위치도 따라온다 (예전엔 손으로 맞춰야 해서 평가 위치의 절반이
# 학습 영역 밖이었다 = 무조건 실패).
# 학습은 연속 랜덤이므로 이 격자점들은 전부 '처음 보는 정확한 좌표'다.
EVAL_GRID_N = 4


def _lin(a, b, n):
    if n == 1:
        return [(a + b) / 2.0]
    return [a + (b - a) * i / (n - 1) for i in range(n)]


EVAL_CUBE_XY_LIST = [
    (round(x, 3), round(y, 3))
    for x in _lin(CUBE_X_RANGE[0], CUBE_X_RANGE[1], EVAL_GRID_N)
    for y in _lin(CUBE_Y_RANGE[0], CUBE_Y_RANGE[1], EVAL_GRID_N)
]

# 평가 성공 판정
SUCCESS_XY_TOL = 0.05
SUCCESS_MIN_LIFT = 0.04


if __name__ == "__main__":
    print(f"이미지        {IMG_W}x{IMG_H}")
    print(f"오버헤드      pos={OVER_TRANSLATE} rot={OVER_ROTATE} focal={OVER_FOCAL}")
    print(f"손목          pos={WRIST_TRANSLATE} rot={WRIST_ROTATE} focal={WRIST_FOCAL}")
    print(f"조명          {LIGHT_INTENSITY}")
    print(f"큐브 영역     x{CUBE_X_RANGE} y{CUBE_Y_RANGE} "
          f"({(CUBE_X_RANGE[1]-CUBE_X_RANGE[0])*100:.0f}x"
          f"{(CUBE_Y_RANGE[1]-CUBE_Y_RANGE[0])*100:.0f}cm)")
    print(f"목표          {TARGET_FIXED_XY} ±{TARGET_JITTER*1000:.0f}mm")
    print(f"평가 위치     {len(EVAL_CUBE_XY_LIST)}개 ({EVAL_GRID_N}x{EVAL_GRID_N} 격자)")
    for i in range(0, len(EVAL_CUBE_XY_LIST), EVAL_GRID_N):
        print("              " + "  ".join(
            f"({x:.3f},{y:.3f})" for x, y in EVAL_CUBE_XY_LIST[i:i + EVAL_GRID_N]))
