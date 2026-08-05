#!/usr/bin/env python3
# export_depth_control.py
# ---------------------------------------------------------------------------
# 수집된 episode_*.npz 의 depth를 에피소드별 16bit PNG 폴더로 빼낸다.
# 이 폴더를 cosmos 쪽 make_control.py 가 --mode depth --depth-dir 로 그대로 읽는다.
#
# 왜 mp4를 바로 만들지 않나
# -----------------------
# 깊이 정규화(에피소드 전체 범위 1회, 근거리=밝게 반전)는 이미 make_control.py가
# 신중하게 하고 있고, 그 로직을 여기 복제하면 두 곳이 어긋난다. 이 스크립트는
# "npz에서 프레임을 꺼내 올바른 인덱스로 나열"하는 것만 한다.
#
# 프레임 인덱스가 왜 중요한가
# -------------------------
# control 비디오는 학습셋 에피소드와 프레임이 1:1로 대응해야 한다. 어긋나면
# 생성된 영상이 액션 라벨과 다른 시점을 가리키고, 증강 데이터 전체가 조용히
# 오염된다(성공률만 떨어지고 원인은 안 보인다).
# 그래서 solve하지 않고 subsample_dataset.py 의 선택 규칙을 그대로 복사한다:
#     idx = range(0, T, stride) + (마지막 프레임이 빠졌으면 T-1 추가)
# 저쪽이 바뀌면 이쪽도 같이 바꿔야 한다.
#
# 사용법
# -----
#   python export_depth_control.py --src /data/jinju/bc_data_v12 \
#                                  --out /data/jinju/depth_ctrl_v12
#
#   # 그다음 cosmos 쪽에서 (에피소드 하나)
#   python make_control.py --mode depth --fps 20 \
#       --depth-dir /data/jinju/depth_ctrl_v12/episode_0000 \
#       --out /data/jinju/ctrl_v12/episode_0000.mp4
#
# Deps: numpy, imageio  (isaacsim 불필요 - 아무 파이썬에서나 돈다)
# ---------------------------------------------------------------------------
import argparse
import glob
import os
import sys

import numpy as np

# stride 기본값은 수집·학습과 같은 곳에서 온다. 어긋나면 프레임이 밀린다.
try:
    from scene_config import TRAIN_STRIDE, IMG_W, IMG_H
except Exception:
    TRAIN_STRIDE, IMG_W, IMG_H = 3, 320, 240


def keep_indices(T, stride):
    """subsample_dataset.py convert() 와 동일한 프레임 선택."""
    idx = list(range(0, T, stride))
    # 마지막 프레임(물체를 놓은 최종 상태)은 항상 남긴다
    if idx[-1] != T - 1:
        idx.append(T - 1)
    return idx


def upscale(d, factor):
    """정수배 nearest 확대. 의존성 없이 정확하다.

    보간(bilinear/lanczos)을 쓰지 않는 이유: 깊이 불연속(큐브 윤곽, 팔 실루엣)을
    가로질러 섞으면 실제로는 없는 중간 깊이의 면이 생긴다. Cosmos는 그 경계를
    기하 구조로 읽으므로, 뭉개진 경계는 그대로 뭉개진 형상으로 나온다.
    """
    if factor == 1:
        return d
    return np.repeat(np.repeat(d, factor, axis=0), factor, axis=1)


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--src", required=True, help="episode_*.npz 가 있는 폴더")
    p.add_argument("--out", required=True, help="에피소드별 PNG 폴더를 만들 위치")
    p.add_argument("--cam", default="over",
                   help="어느 카메라의 depth (기본 over; 수집 시 담은 것만 가능)")
    p.add_argument("--stride", type=int, default=TRAIN_STRIDE,
                   help=f"N프레임마다 1개 (기본 {TRAIN_STRIDE}, scene_config.TRAIN_STRIDE)")
    p.add_argument("--scale", type=int, default=3,
                   help="정수배 확대. 320x240 x3 = 960x720 (Cosmos transfer 기준 해상도)")
    p.add_argument("--limit", type=int, help="앞쪽 N개 에피소드만 (시험용)")
    args = p.parse_args()

    if args.scale < 1:
        p.error("--scale 은 1 이상의 정수")

    files = sorted(glob.glob(os.path.join(args.src, "episode_*.npz")))
    if not files:
        sys.exit(f"episode_*.npz 가 없다: {args.src}")
    if args.limit:
        files = files[:args.limit]

    import imageio.v3 as iio

    key = f"depth_{args.cam}"
    total_frames = 0
    for i, f in enumerate(files):
        d = np.load(f)
        if key not in d.files:
            sys.exit(f"{os.path.basename(f)} 에 '{key}' 가 없다. RECORD_DEPTH=False로 "
                     f"수집된 데이터이거나 --cam 이 틀렸다. 들어있는 키: {list(d.files)}")

        depth = d[key]
        # T는 subsample_dataset.py 와 같은 방식으로 잡는다(가장 짧은 축에 맞춘다).
        lens = [len(d["obs"]), len(d["actions"]), len(depth)]
        if "images_over" in d.files:
            lens.append(len(d["images_over"]))
        if "images_wrist" in d.files:
            lens.append(len(d["images_wrist"]))
        T = min(lens)
        if len(depth) != T:
            print(f"  [!] {os.path.basename(f)}: depth {len(depth)}프레임인데 "
                  f"기준 T={T} - 짧은 쪽에 맞춘다")

        idx = keep_indices(T, args.stride)
        ep_name = os.path.splitext(os.path.basename(f))[0]     # episode_0000
        ep_dir = os.path.join(args.out, ep_name)
        os.makedirs(ep_dir, exist_ok=True)

        for n, t in enumerate(idx):
            frame = np.asarray(depth[t], dtype=np.uint16)
            if frame.ndim == 3:
                frame = frame[..., 0]
            # 16bit 그레이스케일 PNG. make_control.py 의 load_depth_frame 이
            # 이 값을 float32(밀리미터)로 그대로 읽는다.
            iio.imwrite(os.path.join(ep_dir, f"{n:06d}.png"), upscale(frame, args.scale))

        total_frames += len(idx)
        h, w = depth.shape[1] * args.scale, depth.shape[2] * args.scale
        if i % 10 == 0 or i == len(files) - 1:
            print(f"[export] [{i+1}/{len(files)}] {ep_name}: {T} -> {len(idx)}프레임 "
                  f"{w}x{h}  (누적 {total_frames})")

    print(f"\n[export] 완료: 에피소드 {len(files)}개 / {total_frames}프레임 -> {args.out}")
    print("\n[다음] control mp4 만들기 (cosmos 쪽 make_control.py):")
    print(f'  for d in {args.out}/episode_*; do')
    print(f'    python make_control.py --mode depth --fps {60 // args.stride} \\')
    print(f'      --depth-dir "$d" --out "ctrl_$(basename $d).mp4"')
    print(f'  done')
    print("\n[!] 프레임 수가 학습셋 에피소드와 같아야 한다. subsample_dataset.py 를 "
          f"--stride {args.stride} 로 돌린 데이터셋에만 맞는다.")


if __name__ == "__main__":
    sys.exit(main())
