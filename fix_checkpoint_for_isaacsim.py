#!/usr/bin/env python3
# fix_checkpoint_for_isaacsim.py
# ---------------------------------------------------------------------------
# lerobot 0.6.1로 학습한 체크포인트를 Isaac Sim 번들 lerobot(0.4.4)이 읽을 수 있게
# config.json에서 구버전이 모르는 키를 지운다. draccus가 모르는 키를 만나면
# DecodingError로 죽기 때문이다:
#   draccus.utils.DecodingError: The fields `pretrained_revision` are not valid for ACTConfig
#
# 원본은 config.json.orig 로 백업한다(이미 있으면 덮어쓰지 않는다).
#
# 실행:
#   C:\isaacsim\python.bat fix_checkpoint_for_isaacsim.py C:\Users\user\Desktop\act_v11_90k
# ---------------------------------------------------------------------------
import json
import os
import shutil
import sys

# 0.4.4의 ACTConfig에 없는 필드들. 새 키가 추가돼 또 막히면 에러 메시지에 나온
# 필드명을 여기 추가하면 된다.
UNKNOWN_KEYS = ["pretrained_revision"]


def main():
    if len(sys.argv) != 2:
        raise SystemExit("사용법: fix_checkpoint_for_isaacsim.py <체크포인트 폴더>")
    d = sys.argv[1]
    cfg = os.path.join(d, "config.json")
    if not os.path.isfile(cfg):
        raise SystemExit(f"config.json이 없습니다: {cfg}\n"
                         f"  폴더 안에 config.json / model.safetensors 가 있어야 합니다.")

    with open(cfg, encoding="utf-8") as fh:
        data = json.load(fh)

    removed = [k for k in UNKNOWN_KEYS if k in data]
    if not removed:
        print(f"[fix] 지울 키가 없습니다 (이미 처리됨): {cfg}")
    else:
        bak = cfg + ".orig"
        if not os.path.exists(bak):
            shutil.copy2(cfg, bak)
            print(f"[fix] 백업 -> {bak}")
        for k in removed:
            data.pop(k)
        with open(cfg, "w", encoding="utf-8") as fh:
            json.dump(data, fh, indent=2)
        print(f"[fix] 제거: {', '.join(removed)}")

    print(f"[fix] type={data.get('type')} chunk_size={data.get('chunk_size')} "
          f"n_action_steps={data.get('n_action_steps')} "
          f"temporal_ensemble_coeff={data.get('temporal_ensemble_coeff')}")
    feats = list((data.get("input_features") or {}).keys())
    print(f"[fix] 입력: {feats}")
    for f in ("model.safetensors", "policy_preprocessor.json"):
        p = os.path.join(d, f)
        print(f"[fix] {f}: {'있음' if os.path.exists(p) else '없음 (!)'}")
    print(f"\n[fix] 이제 이렇게 띄우면 됩니다:")
    print(f'  set "MODEL_PATH={d}" && C:\\isaacsim\\python.bat -u act_policy_server.py')


if __name__ == "__main__":
    main()
