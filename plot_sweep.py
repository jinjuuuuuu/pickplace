#!/usr/bin/env python3
# plot_sweep.py
# ---------------------------------------------------------------------------
# 체크포인트별 성공률 곡선을 그린다. "과적합인가?"에 답하는 그림이다.
#
# 이 파이프라인에는 검증 셋이 없으므로(lerobot eval_freq=None) train/val 손실
# 격차를 볼 수 없다. 대신 학습 스텝에 대한 태스크 성공률을 그려서 판단한다:
#   - 오르다 마지막이 최고 -> 과적합 신호 없음
#   - 중간 정점 후 하락    -> 그 지점 이후가 과적합
# 표본이 작으면 위아래로 흔들리므로 Wilson 95% 신뢰구간을 함께 그린다.
#
# 실행:
#   # eval_sweep.sh 결과 폴더에서 읽기 (워크스테이션)
#   python plot_sweep.py --dir eval_sweep_act_pickplace_v11 --png sweep.png
#
#   # 이미 아는 숫자로 그리기 ("스텝:성공/전체" 나열)
#   python plot_sweep.py --data "30000:12/16 60000:11/16 90000:16/16 120000:15/16 150000:16/16" \
#       --extra "150000:29/30:랜덤 30개" --png sweep.png
# ---------------------------------------------------------------------------
import argparse
import glob
import json
import math
import os


def wilson(k, n, z=1.96):
    """Wilson score 구간. 16개 중 16개처럼 극단값에서도 정상 동작한다."""
    if n == 0:
        return 0.0, 0.0
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    hw = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return max(0.0, c - hw), min(1.0, c + hw)


def from_dir(d):
    rows = []
    for f in sorted(glob.glob(os.path.join(d, "eval_*.json"))):
        base = os.path.basename(f)[5:-5]
        if not base.isdigit():
            continue
        j = json.load(open(f, encoding="utf-8"))
        rows.append((int(base), j["n_success"], j["n_episodes"]))
    if not rows:
        raise SystemExit(f"eval_*.json 을 못 찾았습니다: {d}")
    return sorted(rows)


def from_data(text):
    rows = []
    for tok in text.split():
        step, frac = tok.split(":")
        k, n = frac.split("/")
        rows.append((int(step), int(k), int(n)))
    return sorted(rows)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dir", default="", help="eval_sweep.sh 결과 폴더")
    p.add_argument("--data", default="", help='"30000:12/16 60000:11/16 ..." 형식')
    p.add_argument("--extra", default="",
                   help='별도 표시할 점. "150000:29/30:라벨" 형식, 공백으로 여러 개')
    p.add_argument("--frames", type=int, default=23900,
                   help="학습 데이터 프레임 수 (에폭 축 계산용)")
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--png", default="sweep.png")
    args = p.parse_args()

    if args.dir:
        rows = from_dir(args.dir)
    elif args.data:
        rows = from_data(args.data)
    else:
        raise SystemExit("--dir 또는 --data 중 하나가 필요합니다")

    extras = []
    for tok in args.extra.split() if args.extra else []:
        parts = tok.split(":")
        step, frac = int(parts[0]), parts[1]
        k, n = (int(v) for v in frac.split("/"))
        extras.append((step, k, n, parts[2] if len(parts) > 2 else ""))

    print(f"{'step':>8} {'에폭':>6} {'성공':>8} {'성공률':>8}  95% CI")
    for step, k, n in rows:
        lo, hi = wilson(k, n)
        ep = step * args.batch / args.frames
        print(f"{step:>8} {ep:>6.1f} {k:>4}/{n:<3} {k/n*100:>7.1f}%  "
              f"[{lo*100:.0f}, {hi*100:.0f}]")
    for step, k, n, lab in extras:
        lo, hi = wilson(k, n)
        print(f"{step:>8} {'':>6} {k:>4}/{n:<3} {k/n*100:>7.1f}%  "
              f"[{lo*100:.0f}, {hi*100:.0f}]  <- {lab}")

    plot(rows, extras, args)


def plot(rows, extras, args):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("[png] matplotlib이 없어 그림은 건너뜁니다")
        return

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
    L = (lambda k, e: k if ko else e)

    steps = [r[0] for r in rows]
    rate = [r[1] / r[2] * 100 for r in rows]
    lo = [rate[i] - wilson(r[1], r[2])[0] * 100 for i, r in enumerate(rows)]
    hi = [wilson(r[1], r[2])[1] * 100 - rate[i] for i, r in enumerate(rows)]

    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    ax.errorbar(steps, rate, yerr=[lo, hi], fmt="o-", lw=2, ms=7,
                capsize=4, c="tab:blue", ecolor="tab:blue", elinewidth=1.2,
                alpha=0.95, label=L(f"격자 {rows[0][2]}개 위치", f"grid, n={rows[0][2]}"),
                zorder=3)

    for step, k, n, lab in extras:
        r = k / n * 100
        l, h = wilson(k, n)
        ax.errorbar([step], [r], yerr=[[r - l * 100], [h * 100 - r]], fmt="s",
                    ms=9, capsize=4, c="tab:red", ecolor="tab:red",
                    elinewidth=1.2, zorder=4,
                    label=(lab or L(f"랜덤 {n}개", f"random, n={n}")) + f" ({k}/{n})")

    best = max(rows, key=lambda r: r[1] / r[2])
    ax.axhline(best[1] / best[2] * 100, ls=":", c="gray", lw=1, zorder=1)

    ax.set_xlabel(L("학습 스텝", "training steps"))
    ax.set_ylabel(L("성공률 [%]", "success rate [%]"))
    ax.set_ylim(0, 108)
    ax.set_xlim(0, max(steps) * 1.08)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=9, loc="lower right")
    ax.set_title(L("체크포인트별 성공률 — 오르다 포화, 하락 없음 = 과적합 신호 없음",
                   "success rate vs training steps"), fontsize=11)

    # 위쪽에 에폭 축
    top = ax.secondary_xaxis(
        "top",
        functions=(lambda s: s * args.batch / args.frames,
                   lambda e: e * args.frames / args.batch))
    top.set_xlabel(L(f"에폭 (프레임 {args.frames:,} / batch {args.batch})",
                     f"epochs ({args.frames:,} frames / batch {args.batch})"), fontsize=9)

    fig.tight_layout()
    fig.savefig(args.png, dpi=160)
    print(f"[png] 저장: {args.png}")


if __name__ == "__main__":
    main()
