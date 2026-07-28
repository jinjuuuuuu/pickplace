#!/usr/bin/env bash
# eval_sweep.sh - 체크포인트별 성공률을 재서 과적합 여부를 본다 (워크스테이션 전용)
# ---------------------------------------------------------------------------
# 왜 필요한가
# ----------
# 이 파이프라인에는 검증 셋이 없다(lerobot eval_freq=None, 수집분 100% 학습).
# 그래서 train/val 손실 격차를 볼 수 없고, "과적합했는가"를 손실로 판단할 수단이
# 없다. 대신 태스크 성공률을 학습 스텝에 대해 그려서 판단한다:
#   - 성공률이 계속 오르다 끝에서 최고 -> 아직 과적합 아님(더 학습해도 됨)
#   - 중간에서 정점을 찍고 내려감      -> 그 지점 이후가 과적합
#   - 처음부터 평평                    -> 학습이 아니라 다른 병목
#
# 실행:
#   conda activate lerobot
#   bash eval_sweep.sh /data/jinju/act_pickplace_v11
#   bash eval_sweep.sh /data/jinju/act_pickplace_v11 "030000 090000 150000"   # 일부만
#
# 분포 밖(학습한 적 없는 영역) 성능을 같이 보려면:
#   EVAL_MARGIN=-0.04 bash eval_sweep.sh /data/jinju/act_pickplace_v11
#
# 결과: eval_sweep_<run>/eval_<step>.json  +  마지막에 요약 표
# ---------------------------------------------------------------------------
set -u

RUN_DIR="${1:?사용법: bash eval_sweep.sh <output_dir> [\"step1 step2 ...\"]}"
RUN_NAME="$(basename "$RUN_DIR")"
CKPT_DIR="$RUN_DIR/checkpoints"

if [ ! -d "$CKPT_DIR" ]; then
  echo "체크포인트 폴더가 없습니다: $CKPT_DIR" >&2
  exit 1
fi

# 인자로 스텝을 주지 않으면 존재하는 체크포인트를 전부 (last 심볼릭 링크는 제외)
if [ $# -ge 2 ]; then
  STEPS="$2"
else
  STEPS="$(cd "$CKPT_DIR" && ls -d [0-9]* 2>/dev/null | sort)"
fi
if [ -z "$STEPS" ]; then
  echo "평가할 체크포인트가 없습니다. ls $CKPT_DIR" >&2
  exit 1
fi

OUT="eval_sweep_${RUN_NAME}"
mkdir -p "$OUT"
echo "[sweep] $RUN_NAME | 체크포인트: $(echo $STEPS | tr '\n' ' ')"
echo "[sweep] 결과 -> $OUT/"
echo "[sweep] EVAL_MARGIN=${EVAL_MARGIN:-기본(0.02, 분포 내부)}"
echo

for S in $STEPS; do
  MP="$CKPT_DIR/$S/pretrained_model"
  if [ ! -d "$MP" ]; then
    echo "[sweep] $S 건너뜀 (없음: $MP)"
    continue
  fi
  echo "===== step $S ====="

  MODEL_PATH="$MP" python act_policy_server.py > "$OUT/server_$S.log" 2>&1 &
  SRV=$!
  # 클라이언트가 최대 60초 재시도하므로 여기서 오래 기다릴 필요는 없다
  sleep 2

  # 시뮬은 Isaac Sim 자체 python으로 돌린다 (conda 환경과 별개)
  /data/isaacsim/python.sh -u eval_act_v5_client.py 2>&1 | tee "$OUT/client_$S.log" | \
      grep -E "^\[client\] (starting|ep|DONE|전송량|손-큐브)"

  kill "$SRV" 2>/dev/null
  wait "$SRV" 2>/dev/null

  if [ -f eval_results_act_v5.json ]; then
    mv eval_results_act_v5.json "$OUT/eval_$S.json"
  else
    echo "[sweep] 경고: step $S 결과 파일이 없습니다. $OUT/client_$S.log 확인"
  fi
  echo
done

echo "===== 요약 ====="
python - "$OUT" <<'PY'
import json, glob, os, sys
out = sys.argv[1]
rows = []
for f in sorted(glob.glob(os.path.join(out, "eval_*.json"))):
    step = int(os.path.basename(f)[5:-5])
    d = json.load(open(f))
    rows.append((step, d))
if not rows:
    print("결과 파일이 없습니다"); raise SystemExit(1)

print(f"{'step':>8} {'에폭':>6} {'성공률':>8} {'성공':>6} {'손-큐브':>8} "
      f"{'닫은지점 표준편차':>18} {'검은프레임':>10}")
for step, d in rows:
    n = d["n_episodes"]
    std = d.get("ee_at_close_std")
    print(f"{step:>8} {'':>6} {d['success_rate']*100:>7.1f}% "
          f"{d['n_success']:>3}/{n:<3} {d.get('avg_min_ee_cube', 0):>8.4f} "
          f"{str(std):>18} {d.get('blank_camera_frames', '?'):>10}")

best = max(rows, key=lambda r: r[1]["success_rate"])
last = rows[-1]
print()
print(f"최고: step {best[0]} ({best[1]['success_rate']*100:.1f}%)")
print(f"마지막: step {last[0]} ({last[1]['success_rate']*100:.1f}%)")
if best[0] == last[0]:
    print("-> 마지막이 최고다. 과적합 신호 없음 (더 학습해도 될 여지가 있다)")
elif best[1]["success_rate"] - last[1]["success_rate"] >= 0.1:
    print(f"-> step {best[0]} 이후 {(best[1]['success_rate']-last[1]['success_rate'])*100:.0f}%p "
          f"하락. **과적합**으로 보인다. step {best[0]} 체크포인트를 쓸 것")
else:
    print("-> 정점 이후 하락이 10%p 미만이라 노이즈와 구분이 안 된다. "
          "EVAL_GRID_N=6으로 표본을 늘려 재확인할 것")
PY
