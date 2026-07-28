#!/usr/bin/env python3
# diag_grasp_timing.py
# ---------------------------------------------------------------------------
# "정책이 그리퍼를 너무 일찍 닫는다"를 검증/반증한다.
#
# 왜 필요한가
# ----------
# 평가에서 16/16 에피소드가 "손-큐브 최소거리는 1~9mm인데 그리퍼를 닫은 지점은
# 2~10cm 떨어져 있다"로 나왔다. 이걸 보고 조기 폐쇄라고 판단했지만, **시연이
# 실제로 몇 cm에서 닫는지는 재본 적이 없다.** franka.end_effector는 손바닥
# (panda_hand) 프레임이고 손가락 끝보다 위에 있어서, 접근 각도에 따라 손바닥
# 중심과 큐브의 XY가 원래 수 cm 벌어질 수 있다. 시연도 그렇다면 조기 폐쇄가
# 아니라 정상이고, 실패 원인은 다른 곳이다.
#
# obs 레이아웃(수집 스크립트 get_observation):
#   [joint_pos(9), joint_vel(9), cube_rel(3), target_rel(3), cube_pos(3)]
#   cube_rel = cube_pos - ee_pos  <- 평가의 (cube - ee)와 같은 양
#
# 실행:  python diag_grasp_timing.py --src /data/jinju/bc_data_v11
# ---------------------------------------------------------------------------
import os
import glob
import argparse

import numpy as np

from scene_config import GRIPPER_CLOSING_RAW_THRESH

CUBE_REL = slice(18, 21)
GRIPPER_IDX = 7
CLOSE_THRESH = GRIPPER_CLOSING_RAW_THRESH


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--src", default="/data/jinju/bc_data_v11")
    p.add_argument("--n", type=int, default=0, help="앞 N개만 (0=전부)")
    args = p.parse_args()

    files = sorted(glob.glob(os.path.join(args.src, "episode_*.npz")))
    if not files:
        raise SystemExit(f"에피소드가 없습니다: {args.src}")
    if args.n:
        files = files[:args.n]

    print(f"[diag] {args.src}  에피소드 {len(files)}개\n")
    print("  ep   닫은프레임  닫는순간 cube-ee (m)              전체 최소")
    print("                    dx      dy      dz    |xy|      |xy|")
    print("  ---  ---------  ------  ------  ------  ------    ------")

    at_close, min_xy, close_frac = [], [], []
    for i, f in enumerate(files):
        d = np.load(f)
        obs = np.asarray(d["obs"], dtype=np.float32)
        act = np.asarray(d["actions"], dtype=np.float32)
        T = min(len(obs), len(act))
        rel = obs[:T, CUBE_REL]
        xy = np.linalg.norm(rel[:, :2], axis=1)

        closed = np.where(act[:T, GRIPPER_IDX] < CLOSE_THRESH)[0]
        if len(closed) == 0:
            print(f"  {i:3d}  (그리퍼를 닫은 프레임이 없음)")
            continue
        t = int(closed[0])
        at_close.append(rel[t])
        min_xy.append(float(xy.min()))
        close_frac.append(t / T)

        if i < 12 or i == len(files) - 1:
            print(f"  {i:3d}  {t:5d}/{T:<4d} {rel[t,0]:+7.4f} {rel[t,1]:+7.4f} "
                  f"{rel[t,2]:+7.4f}  {np.linalg.norm(rel[t,:2]):6.4f}    {xy.min():6.4f}")
        elif i == 12:
            print("  ...")

    if not at_close:
        raise SystemExit("[diag] 닫는 프레임을 못 찾았습니다")

    a = np.array(at_close)
    axy = np.linalg.norm(a[:, :2], axis=1)
    print(f"\n[diag] 시연 {len(a)}개 요약 — 그리퍼를 닫는 순간의 cube - ee")
    print(f"    dx  평균 {a[:,0].mean():+.4f}  표준편차 {a[:,0].std():.4f}")
    print(f"    dy  평균 {a[:,1].mean():+.4f}  표준편차 {a[:,1].std():.4f}")
    print(f"    dz  평균 {a[:,2].mean():+.4f}  표준편차 {a[:,2].std():.4f}")
    print(f"    |xy| 평균 {axy.mean():.4f}  (최소 {axy.min():.4f}  최대 {axy.max():.4f})")
    print(f"    에피소드 전체 최소 |xy| 평균 {np.mean(min_xy):.4f}")
    print(f"    닫는 시점: 에피소드의 {np.mean(close_frac)*100:.1f}% 지점")

    print(f"\n[diag] 해석")
    print(f"    평가에서 측정된 '닫은 지점의 |xy|'는 평균 0.031m 였다.")
    if axy.mean() > 0.02:
        print(f"    시연도 {axy.mean():.3f}m 에서 닫는다 -> **조기 폐쇄가 아니다.**")
        print(f"    손바닥 프레임과 손가락 끝의 오프셋 때문이고, 정책은 시연을")
        print(f"    제대로 따라하고 있다. 실패 원인은 XY 타이밍이 아니라 다른 곳")
        print(f"    (dz 높이, 접근 자세, 그리퍼 힘 등)에 있다.")
        print(f"    -> 아래 dz를 평가값과 비교할 것. dz 평균 {a[:,2].mean():+.4f}m")
    else:
        print(f"    시연은 {axy.mean():.3f}m 에서 닫는다 (평가 0.031m 보다 훨씬 가깝다)")
        print(f"    -> **조기 폐쇄가 맞다.** 정책이 시연보다 먼 곳에서 닫고 있다.")


if __name__ == "__main__":
    main()
