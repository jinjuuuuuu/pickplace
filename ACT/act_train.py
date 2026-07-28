# act_train.py  —  ACT 학습 (일반 Python + GPU, Isaac Sim 불필요)
# ---------------------------------------------------------------------------
# act_collect_isaac.py 가 만든 HDF5 데이터로 ACT 정책을 학습한다.
# 원본 ACT(tonyzhaozh/act) / ACTfranka 의 학습 루프를 config.py 로 구동.
#
# 실행 (CUDA GPU 권장):
#   python act_train.py
# 산출물 (CHECKPOINT_DIR):
#   - dataset_stats.pkl   : qpos/action 정규화 통계 (배포에 필요)
#   - policy_best.ckpt     : 검증 손실 최저 가중치
#   - policy_last.ckpt     : 마지막 가중치
# ---------------------------------------------------------------------------
import os, sys, glob, pickle
from copy import deepcopy
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config as C
# detr 모델 빌드시 argparse가 sys.argv를 읽으므로, 빌드 전에 argv를 비운다.
_argv_backup = sys.argv[:]
sys.argv = [sys.argv[0]]

from act_dataset import load_data, make_policy, make_optimizer, set_seed, \
                        compute_dict_mean, detach_dict


def forward_pass(data, policy, device):
    image_data, qpos_data, action_data, is_pad = data
    image_data  = image_data.to(device)
    qpos_data   = qpos_data.float().to(device)
    action_data = action_data.float().to(device)
    is_pad      = is_pad.to(device)
    return policy(qpos_data, image_data, action_data, is_pad)


def main():
    device = C.device
    set_seed(C.SEED)
    os.makedirs(C.CHECKPOINT_DIR, exist_ok=True)

    n_ep = len(glob.glob(os.path.join(C.DATASET_DIR, "episode_*.npz")))
    if n_ep == 0:
        print(f"[train] ❌ 데이터가 없습니다: {C.DATASET_DIR}\n"
              f"        먼저 act_collect_isaac.py 로 데이터를 수집하세요.")
        sys.exit(1)
    print(f"[train] device={device} | 에피소드 {n_ep}개 | "
          f"chunk={C.CHUNK_SIZE} | cams={C.CAMERA_NAMES}")

    train_loader, val_loader, stats, _ = load_data(
        C.DATASET_DIR, n_ep, C.CAMERA_NAMES, C.BATCH_SIZE, C.BATCH_SIZE)

    # 정규화 통계 저장 (배포에서 동일하게 사용)
    with open(os.path.join(C.CHECKPOINT_DIR, "dataset_stats.pkl"), "wb") as f:
        pickle.dump(stats, f)
    print(f"[train] dataset_stats.pkl 저장")

    policy = make_policy("ACT", C.build_policy_config()).to(device)
    optimizer = make_optimizer("ACT", policy)
    print(f"[train] 파라미터 수: {sum(p.numel() for p in policy.parameters()):,}")

    best_val = np.inf
    best_state = None
    for epoch in range(C.NUM_EPOCHS):
        # ---- validation ----
        policy.eval()
        with torch.inference_mode():
            vdicts = [forward_pass(d, policy, device) for d in val_loader]
        vsum = compute_dict_mean(vdicts)
        vloss = vsum["loss"].item()
        if vloss < best_val:
            best_val = vloss
            best_state = deepcopy(policy.state_dict())

        # ---- train ----
        policy.train()
        tdicts = []
        for data in train_loader:
            fd = forward_pass(data, policy, device)
            fd["loss"].backward()
            optimizer.step(); optimizer.zero_grad()
            tdicts.append(detach_dict(fd))
        tsum = compute_dict_mean(tdicts)

        if epoch % 50 == 0 or epoch == C.NUM_EPOCHS - 1:
            print(f"[train] epoch {epoch:4d} | "
                  f"train loss {tsum['loss'].item():.4f} "
                  f"(l1 {tsum['l1'].item():.4f}, kl {tsum['kl'].item():.4f}) | "
                  f"val loss {vloss:.4f} | best {best_val:.4f}")
            torch.save(policy.state_dict(),
                       os.path.join(C.CHECKPOINT_DIR, "policy_last.ckpt"))
            if best_state is not None:
                torch.save(best_state,
                           os.path.join(C.CHECKPOINT_DIR, "policy_best.ckpt"))

    torch.save(policy.state_dict(), os.path.join(C.CHECKPOINT_DIR, "policy_last.ckpt"))
    if best_state is not None:
        torch.save(best_state, os.path.join(C.CHECKPOINT_DIR, "policy_best.ckpt"))
    print(f"\n[train] 완료! best val loss {best_val:.4f}")
    print(f"[train] 저장 위치: {C.CHECKPOINT_DIR}")


if __name__ == "__main__":
    main()
    sys.argv = _argv_backup
