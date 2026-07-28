#!/usr/bin/env python3
# show_eval_positions.py
# ---------------------------------------------------------------------------
# 평가 좌표를 시드별로 나열/그림으로 보여준다 (보고서·발표자료용).
#
# 평가 위치가 어떻게 정해지는지 설명할 때 쓴다. scene_config와 완전히 같은 로직을
# 쓰므로(같은 함수를 import) 여기 나온 좌표가 실제 평가에 쓰이는 좌표다.
#
# 실행:
#   python show_eval_positions.py                      # 시드 0~3, 각 30개
#   python show_eval_positions.py --seeds 0 1 2 3 4    # 시드 지정
#   python show_eval_positions.py --n 16               # 개수 지정
#   python show_eval_positions.py --grid               # 격자(기본 평가) 좌표도 함께
#   python show_eval_positions.py --png positions.png  # 산점도 저장
# ---------------------------------------------------------------------------
import argparse
import random

import scene_config as sc


def sample(n, seed, margin=None):
    """scene_config.EVAL_RANDOM 과 동일한 방식으로 n개를 뽑는다."""
    m = sc.EVAL_MARGIN if margin is None else margin
    rng = random.Random(seed)
    xlo, xhi = sc.CUBE_X_RANGE[0] + m, sc.CUBE_X_RANGE[1] - m
    ylo, yhi = sc.CUBE_Y_RANGE[0] + m, sc.CUBE_Y_RANGE[1] - m
    pts = []
    while len(pts) < n:
        x, y = rng.uniform(xlo, xhi), rng.uniform(ylo, yhi)
        if sc._too_close_to_target(x, y):
            continue
        pts.append((round(x, 4), round(y, 4)))
    return pts


def load_collected(src):
    """수집 좌표를 읽는다. 폴더면 npz의 cube_pos를, .json이면 좌표 목록을 읽는다.

    데이터가 다른 머신에 있을 때는 그쪽에서 좌표만 뽑아 json으로 옮기면 된다:
      python -c "import glob,json,numpy as np; json.dump([[float(np.load(f)['cube_pos'][0]),
        float(np.load(f)['cube_pos'][1])] for f in sorted(glob.glob('DIR/episode_*.npz'))],
        open('collected.json','w'))"
    """
    import glob
    import os
    import numpy as np

    if src.lower().endswith(".json"):
        import json
        with open(src, encoding="utf-8") as fh:
            raw = json.load(fh)
        return [(round(float(p[0]), 4), round(float(p[1]), 4)) for p in raw]

    files = sorted(glob.glob(os.path.join(src, "episode_*.npz")))
    if not files:
        raise SystemExit(f"에피소드가 없습니다: {src}")
    pts = []
    for f in files:
        d = np.load(f)                       # 압축 npz는 요청한 키만 풀린다
        if "cube_pos" not in d.files:
            raise SystemExit(f"cube_pos가 없습니다: {f} (keys={d.files})")
        c = np.asarray(d["cube_pos"], dtype=float).reshape(-1)
        pts.append((round(float(c[0]), 4), round(float(c[1]), 4)))
    return pts


def report_density(pts, args):
    """평가 좌표에서 가장 가까운 학습 좌표까지의 거리 = '얼마나 새로운 위치인가'."""
    import math
    w = (sc.CUBE_X_RANGE[1] - sc.CUBE_X_RANGE[0]) * 100
    h = (sc.CUBE_Y_RANGE[1] - sc.CUBE_Y_RANGE[0]) * 100
    print(f"[수집] 에피소드 {len(pts)}개 | 영역 {w:.0f}x{h:.0f}cm "
          f"| 밀도 {len(pts)/(w*h):.4f}개/cm^2")

    def nn(q, pool):
        return min(math.hypot(q[0] - p[0], q[1] - p[1]) for p in pool)

    # 학습 좌표끼리의 최근접 거리 (수집이 얼마나 촘촘한가)
    if len(pts) > 1:
        own = [min(math.hypot(a[0]-b[0], a[1]-b[1])
                   for j, b in enumerate(pts) if j != i) for i, a in enumerate(pts)]
        own.sort()
        print(f"[수집] 학습 좌표 간 최근접거리: 중간값 {own[len(own)//2]*100:.2f}cm "
              f"(최소 {own[0]*100:.2f} 최대 {own[-1]*100:.2f})")

    # 평가 좌표에서 가장 가까운 학습 좌표까지 — 평가가 얼마나 '처음 보는' 위치인가
    for name, ev in (("격자 4x4", list(sc.EVAL_CUBE_XY_LIST)),
                     (f"랜덤 seed 0 (n={args.n})", sample(args.n, 0))):
        ds = sorted(nn(q, pts) for q in ev)
        print(f"[평가] {name}: 최근접 학습좌표까지 중간값 {ds[len(ds)//2]*100:.2f}cm "
              f"(최소 {ds[0]*100:.2f} 최대 {ds[-1]*100:.2f})")
    print()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3])
    p.add_argument("--n", type=int, default=30)
    p.add_argument("--grid", action="store_true", help="기본 4x4 격자도 함께 표시")
    p.add_argument("--collected", default="",
                   help="수집 데이터 폴더 (예: /data/jinju/bc_data_v11). "
                        "npz의 cube_pos를 읽어 실제 학습 좌표를 별도 패널로 그린다")
    p.add_argument("--overlay", action="store_true",
                   help="수집 좌표(파랑)와 평가 좌표(빨강)를 한 패널에 겹쳐 그린다 "
                        "(--collected 와 함께 쓸 때만 의미 있음)")
    p.add_argument("--png", default="", help="산점도를 저장할 파일 경로")
    p.add_argument("--cols", type=int, default=5, help="좌표를 한 줄에 몇 개씩 찍을지")
    args = p.parse_args()

    w = (sc.CUBE_X_RANGE[1] - sc.CUBE_X_RANGE[0]) * 100
    h = (sc.CUBE_Y_RANGE[1] - sc.CUBE_Y_RANGE[0]) * 100
    print(f"수집 영역   x{sc.CUBE_X_RANGE} y{sc.CUBE_Y_RANGE}  ({w:.0f}x{h:.0f}cm)")
    print(f"평가 영역   경계에서 {sc.EVAL_MARGIN*100:.0f}cm 안쪽 "
          f"(x {sc.CUBE_X_RANGE[0]+sc.EVAL_MARGIN:.3f}~{sc.CUBE_X_RANGE[1]-sc.EVAL_MARGIN:.3f}, "
          f"y {sc.CUBE_Y_RANGE[0]+sc.EVAL_MARGIN:.3f}~{sc.CUBE_Y_RANGE[1]-sc.EVAL_MARGIN:.3f})")
    print(f"목표        {sc.TARGET_FIXED_XY}  (MIN_DISTANCE {sc.MIN_DISTANCE}m 이내는 재추첨)")
    print()

    BLUE, RED = "tab:blue", "tab:red"
    collected = load_collected(args.collected) if args.collected else None

    sets = {}          # 이름 -> [(좌표목록, 색, 범례이름), ...]
    if collected is not None:
        sets[f"수집 데이터 ({len(collected)}ep)"] = [(collected, BLUE, "수집")]
        report_density(collected, args)
    if args.overlay and collected is not None:
        for sd in args.seeds:
            sets[f"수집 + seed {sd}"] = [(collected, BLUE, "수집"),
                                          (sample(args.n, sd), RED, f"평가 seed {sd}")]
    if args.grid:
        sets["격자 4x4"] = [(list(sc.EVAL_CUBE_XY_LIST), RED, "평가")]
    if not args.overlay:
        for sd in args.seeds:
            sets[f"seed {sd}"] = [(sample(args.n, sd), RED, f"평가 seed {sd}")]

    for name, layers in sets.items():
        for pts, _c, lab in layers:
            print(f"[{name}] {lab}  {len(pts)}개")
            for i in range(0, len(pts), args.cols):
                print("   " + "  ".join(f"({x:.3f},{y:.3f})" for x, y in pts[i:i + args.cols]))
        print()

    if args.png:
        save_png(sets, args.png)


def save_png(sets, path):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.patches import Rectangle
    except ImportError:
        print("[png] matplotlib이 없어 그림은 건너뜁니다")
        return

    # 한글 라벨이 두부(□)로 안 나오게 폰트를 잡는다. 없으면 영문 라벨로 떨어진다.
    ko = True
    for f in ("Malgun Gothic", "AppleGothic", "NanumGothic", "Noto Sans CJK KR"):
        try:
            matplotlib.font_manager.findfont(f, fallback_to_default=False)
            matplotlib.rcParams["font.family"] = f
            matplotlib.rcParams["axes.unicode_minus"] = False
            break
        except Exception:
            continue
    else:
        ko = False
        print("[png] 한글 폰트를 못 찾아 영문 라벨로 그립니다")
    _lbl = (lambda k, e: k if ko else e)

    n = len(sets)
    cols = min(n, 5)
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(3.4 * cols, 3.4 * rows), squeeze=False)

    for ax, (name, layers) in zip(axes.flat, sets.items()):
        # 수집 영역
        ax.add_patch(Rectangle((sc.CUBE_X_RANGE[0], sc.CUBE_Y_RANGE[0]),
                               sc.CUBE_X_RANGE[1] - sc.CUBE_X_RANGE[0],
                               sc.CUBE_Y_RANGE[1] - sc.CUBE_Y_RANGE[0],
                               fill=False, ls="--", lw=1.2, ec="tab:blue"))
        # 평가 영역
        m = sc.EVAL_MARGIN
        ax.add_patch(Rectangle((sc.CUBE_X_RANGE[0] + m, sc.CUBE_Y_RANGE[0] + m),
                               (sc.CUBE_X_RANGE[1] - m) - (sc.CUBE_X_RANGE[0] + m),
                               (sc.CUBE_Y_RANGE[1] - m) - (sc.CUBE_Y_RANGE[0] + m),
                               fill=False, lw=1.0, ec="tab:gray"))
        for pts, col, lab in layers:
            ax.scatter([p[0] for p in pts], [p[1] for p in pts], s=26,
                       c=col, alpha=0.75 if len(layers) > 1 else 1.0,
                       zorder=3, label=f"{lab} n={len(pts)}")
        ax.scatter([sc.TARGET_FIXED_XY[0]], [sc.TARGET_FIXED_XY[1]], marker="*",
                   s=170, c="tab:green", zorder=4)
        if len(layers) > 1:
            ax.legend(fontsize=7, loc="lower left", framealpha=0.9)
        ax.set_title(name, fontsize=10)
        ax.set_xlim(0.27, 0.59)
        ax.set_ylim(-0.20, 0.31)
        ax.set_aspect("equal")
        ax.grid(alpha=0.25)
        ax.set_xlabel("x [m]", fontsize=8)
        ax.set_ylabel("y [m]", fontsize=8)
        ax.tick_params(labelsize=7)

    for ax in axes.flat[len(sets):]:
        ax.axis("off")

    fig.tight_layout()
    fig.savefig(path, dpi=160)
    print(f"[png] 저장: {path}")
    print(_lbl("[png] 빨간 점=평가 위치, 초록 별=목표, 파란 점선=수집 영역, 회색 실선=평가 영역",
               "[png] red=eval positions, green star=target, blue dashed=collection region, gray=eval region"))


if __name__ == "__main__":
    main()
