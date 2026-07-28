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


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3])
    p.add_argument("--n", type=int, default=30)
    p.add_argument("--grid", action="store_true", help="기본 4x4 격자도 함께 표시")
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

    sets = {}
    if args.grid:
        sets["격자 4x4"] = list(sc.EVAL_CUBE_XY_LIST)
    for s in args.seeds:
        sets[f"seed {s}"] = sample(args.n, s)

    for name, pts in sets.items():
        print(f"[{name}]  {len(pts)}개")
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

    for ax, (name, pts) in zip(axes.flat, sets.items()):
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
        ax.scatter([p[0] for p in pts], [p[1] for p in pts], s=26,
                   c="tab:red", zorder=3)
        ax.scatter([sc.TARGET_FIXED_XY[0]], [sc.TARGET_FIXED_XY[1]], marker="*",
                   s=170, c="tab:green", zorder=4)
        ax.set_title(f"{name} (n={len(pts)})", fontsize=10)
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
