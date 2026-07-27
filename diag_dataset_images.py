#!/usr/bin/env python3
# diag_dataset_images.py
# ---------------------------------------------------------------------------
# 학습 데이터(bc_data_v9의 episode_*.npz)의 이미지가 "큐브를 볼 수 있는" 이미지인지
# 숫자로 확인한다. 정책이 실패할 때 원인이 (a) 학습량 (b) 도메인 갭 (c) 애초에
# 입력에 정보가 없음 중 무엇인지 가르는 데 쓴다.
#
# 하는 일:
#   1) 이미지 mean/std  -> 평가 하네스의 [client] img ... 출력과 비교(조명 일치 확인)
#   2) 빨간 큐브의 픽셀 수와 bbox -> 큐브가 몇 픽셀로 보이는지
#   3) PIL이 있으면 프레임을 8배 확대해 PNG로 저장 -> 눈으로 확인
#
# 실행:  conda activate lerobot && python diag_dataset_images.py
#        (numpy만 있으면 1,2는 동작한다. Isaac Sim python으로도 실행 가능)
# ---------------------------------------------------------------------------
import os
import glob
import argparse

import numpy as np

# 큐브 색은 (0.8, 0.2, 0.1) 빨강. "R이 G,B보다 확실히 크다"로 큐브 픽셀을 센다.
# 로봇 본체는 흰색/검정, 바닥은 회색이라 이 조건에 걸리지 않는다.
RED_MARGIN = 40


def red_mask(img):
    r = img[:, :, 0].astype(np.int16)
    g = img[:, :, 1].astype(np.int16)
    b = img[:, :, 2].astype(np.int16)
    return (r - np.maximum(g, b)) > RED_MARGIN


def describe(name, img):
    m = red_mask(img)
    n = int(m.sum())
    H, W = img.shape[:2]
    line = (f"    {name:5s} {W}x{H}  mean={img.mean():6.2f} std={img.std():5.2f}  "
            f"빨강 {n:5d}px ({n / (H * W) * 100:5.2f}%)")
    if n:
        ys, xs = np.where(m)
        h, w = ys.max() - ys.min() + 1, xs.max() - xs.min() + 1
        line += f"  bbox {w}x{h}px  중심=({xs.mean():.0f},{ys.mean():.0f})"
    else:
        line += "  bbox 없음 (큐브가 안 보인다!)"
    print(line)
    return n


def save_png(img, path, scale=8):
    try:
        from PIL import Image
    except ImportError:
        return False
    H, W = img.shape[:2]
    Image.fromarray(img).resize((W * scale, H * scale), Image.NEAREST).save(path)
    return True


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--src", default="/data/jinju/bc_data_v9")
    p.add_argument("--n-episodes", type=int, default=3)
    p.add_argument("--frames", default="0,60,120,240",
                   help="확인할 프레임 인덱스(원본 60Hz 기준)")
    p.add_argument("--outdir", default="", help="PNG 저장 폴더(비우면 저장 안 함)")
    args = p.parse_args()

    files = sorted(glob.glob(os.path.join(args.src, "episode_*.npz")))
    if not files:
        raise SystemExit(f"에피소드가 없습니다: {args.src}")
    frames = [int(v) for v in args.frames.split(",")]
    if args.outdir:
        os.makedirs(args.outdir, exist_ok=True)

    print(f"[diag] {args.src}  에피소드 {len(files)}개 중 앞 {args.n_episodes}개 확인\n")

    over_px = []
    for f in files[:args.n_episodes]:
        d = np.load(f)
        iw, io = d["images_wrist"], d["images_over"]
        print(f"  {os.path.basename(f)}  ({len(io)} frames)")
        for t in frames:
            if t >= len(io):
                continue
            print(f"   t={t}")
            describe("wrist", iw[t])
            over_px.append(describe("over", io[t]))
            if args.outdir:
                ok = save_png(iw[t], os.path.join(args.outdir, f"{os.path.basename(f)}_t{t}_wrist.png"))
                save_png(io[t], os.path.join(args.outdir, f"{os.path.basename(f)}_t{t}_over.png"))
                if not ok:
                    print("    (PIL이 없어 PNG 저장을 건너뜀)")
                    args.outdir = ""
        print()

    if over_px:
        med = float(np.median(over_px))
        side = med ** 0.5
        print(f"[diag] 오버헤드에서 큐브는 중간값 {med:.0f}px = 약 {side:.1f}x{side:.1f}px 로 보인다.")
        print("[diag] 참고: ALOHA/ACT 표준은 480x640이고, resnet18의 최종 특징맵은")
        print("       입력 120x160 -> 4x5 (토큰 20개), 480x640 -> 15x20 (토큰 300개)다.")
        if side < 12:
            print("[diag] ⚠ 큐브가 10px 미만이면 ACT가 위치를 1cm 정밀도로 회귀하기 어렵다.")
            print("       해상도를 올리거나(320x240 이상) 오버헤드 화각을 작업영역으로 좁힐 것")
            print("       (focal 24 -> 48mm면 보는 범위가 절반, 큐브 픽셀은 2배).")


if __name__ == "__main__":
    main()
