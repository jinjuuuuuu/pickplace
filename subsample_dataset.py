#!/usr/bin/env python3
# subsample_dataset.py
# ---------------------------------------------------------------------------
# bc_data_v5의 episode_*.npz를 N프레임마다 1개씩 솎아서 LeRobotDataset으로 변환한다.
# 원본 npz는 건드리지 않고, 중간 사본도 만들지 않는다(변환하면서 바로 건너뛴다).
#
# 1) 먼저 stride를 정한다 (lerobot 없이 어디서든 실행 가능):
#      python subsample_dataset.py --analyze --src bc_data_v5
#
# 2) 워크스테이션 conda 'lerobot' 환경에서 변환:
#      conda activate lerobot
#      python subsample_dataset.py --src /data/jinju/bc_data_v5 --stride 3 \
#             --repo-id jamongsteak/pickplace_vision_v6_s3
#
# 배포 시 주의: stride N으로 학습한 정책은 (원본fps/N)Hz로 판단한다.
#              시뮬레이터는 여전히 원본fps로 돌므로 액션 하나를 N스텝 유지해야 한다.
#              (자세한 내용은 --analyze 출력 마지막의 안내 참조)
# ---------------------------------------------------------------------------
import os
import glob
import argparse

import numpy as np

# stride 기본값은 scene_config에서 온다. 평가의 ACTION_REPEAT도 같은 값을 읽으므로
# stride를 바꿀 때 한 곳만 고치면 된다 (어긋나면 로봇이 학습 때와 다른 속도로
# 움직이는데, 에러가 안 나고 그냥 실패한다).
try:
    from scene_config import TRAIN_STRIDE, RECORD_HZ
except ImportError:      # 이 스크립트만 따로 복사해 쓰는 경우
    TRAIN_STRIDE, RECORD_HZ = 3, 60

# obs 레이아웃: [joint_pos(9), joint_vel(9), cube_rel(3), target_rel(3), cube_pos(3)]
PROPRIO_DIM = 9
ARM_DOF = 7          # 그리퍼 2축을 뺀 팔 관절
GRIPPER_IDX = 7      # action[7:9]가 그리퍼
TASK = "pick up the cube and place it on the target"


def find_episodes(src):
    files = sorted(glob.glob(os.path.join(src, "episode_*.npz")))
    if not files:
        raise SystemExit(f"에피소드 파일이 없습니다: {src}")
    return files


# ---------------------------------------------------------------------------
# 분석 모드: stride를 얼마로 잡을지 실제 데이터로 판단한다
# ---------------------------------------------------------------------------
def analyze(files, src_fps, chunk_size, sample_n):
    files = files[:sample_n]
    print(f"[analyze] {src_fps}Hz 기록 가정 | 에피소드 {len(files)}개 표본\n")

    lengths, arm_q, gripper = [], [], []
    for f in files:
        d = np.load(f)
        obs = np.asarray(d["obs"], dtype=np.float32)
        act = np.asarray(d["actions"], dtype=np.float32)
        T = min(len(obs), len(act))
        lengths.append(T)
        arm_q.append(obs[:T, :ARM_DOF])
        gripper.append(act[:T, GRIPPER_IDX])

    lengths = np.array(lengths)
    print(f"  프레임/에피소드: min={lengths.min()} max={lengths.max()} "
          f"mean={lengths.mean():.0f}  ({lengths.mean()/src_fps:.1f}초)")

    # 전체 에피소드 수로 환산한 총 프레임 (표본이 아니라 원본 전체 기준)
    print()
    print("  stride | 프레임/ep |   Hz  | 스텝당 관절변화(rad)      | chunk=%d 지평" % chunk_size)
    print("         |           |       |  평균      95%      최대   | (초 / 에피소드 대비)")
    print("  -------+-----------+-------+--------------------------+----------------------")

    for s in range(1, 9):
        deltas = np.concatenate([np.abs(np.diff(q[::s], axis=0)).max(axis=1) for q in arm_q])
        kept = int(np.ceil(lengths.mean() / s))
        hz = src_fps / s
        horizon_s = chunk_size * s / src_fps
        horizon_frac = min(1.0, chunk_size * s / lengths.mean())
        print(f"    {s}    |   {kept:5d}   | {hz:5.1f} | "
              f"{deltas.mean():.4f}  {np.percentile(deltas,95):.4f}  {deltas.max():.4f} | "
              f"{horizon_s:5.1f}s / {horizon_frac*100:5.1f}%")

    # 그리퍼 개폐 시점이 솎기로 얼마나 흔들리는지
    print()
    n_switch = [int((np.diff(g) != 0).sum()) for g in gripper]
    print(f"  그리퍼 전환 횟수/에피소드: {min(n_switch)}~{max(n_switch)} "
          f"(솎으면 전환 시점이 최대 stride/{src_fps}초만큼 밀림 - "
          f"stride 3이면 {3/src_fps*1000:.0f}ms)")

    print()
    print("  [읽는 법]")
    print("   - 스텝당 관절변화가 너무 작으면(<=0.005 rad) 연속 프레임이 사실상 중복이다.")
    print("   - 너무 크면(>=0.05 rad) 정책이 배우기 어려운 급격한 점프가 된다.")
    print("   - chunk 지평이 에피소드의 30% 이상은 되어야 정책이 태스크 흐름을 본다.")
    print()
    print(f"  [배포] stride N으로 학습하면 정책은 (원본fps/N)Hz로 판단한다.")
    print(f"         추론 루프에서 액션 하나를 N번 유지할 것:")
    print(f"             if step % N == 0: action = policy.select_action(obs)")
    print(f"             apply(action)")


# ---------------------------------------------------------------------------
# 변환 모드
# ---------------------------------------------------------------------------
def convert(files, args):
    try:
        from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
    except ImportError:
        from lerobot.datasets.lerobot_dataset import LeRobotDataset

    s = args.stride
    if args.src_fps % s != 0:
        print(f"[warn] {args.src_fps} / {s} 가 정수가 아님 -> fps를 반올림해서 기록한다")
    out_fps = max(1, round(args.src_fps / s))

    d0 = np.load(files[0])
    if "images_wrist" not in d0.files or "images_over" not in d0.files:
        raise SystemExit(f"이미지가 없는 데이터입니다. keys={d0.files}")

    ACT_DIM = int(d0["actions"].shape[1])
    H_W, W_W = int(d0["images_wrist"].shape[1]), int(d0["images_wrist"].shape[2])
    H_O, W_O = int(d0["images_over"].shape[1]), int(d0["images_over"].shape[2])

    print(f"[convert] 에피소드 {len(files)}개 | stride={s} | "
          f"{args.src_fps}Hz -> {out_fps}Hz | state={PROPRIO_DIM} act={ACT_DIM} | "
          f"wrist={H_W}x{W_W} over={H_O}x{W_O}")
    print(f"[convert] repo_id={args.repo_id}")

    features = {
        "observation.images.wrist": {"dtype": "video" if args.videos else "image",
                                     "shape": (H_W, W_W, 3),
                                     "names": ["height", "width", "channels"]},
        "observation.images.over": {"dtype": "video" if args.videos else "image",
                                    "shape": (H_O, W_O, 3),
                                    "names": ["height", "width", "channels"]},
        "observation.state": {"dtype": "float32", "shape": (PROPRIO_DIM,), "names": None},
        "action": {"dtype": "float32", "shape": (ACT_DIM,), "names": None},
    }

    dataset = LeRobotDataset.create(
        repo_id=args.repo_id,
        fps=out_fps,
        features=features,
        robot_type="franka",
        use_videos=args.videos,
    )

    # add_frame / save_episode 시그니처가 lerobot 버전마다 달라서 첫 호출에서 판별
    _mode = {"v": None}

    def add_frame(frame, task):
        m = _mode["v"]
        if m == "kw":
            return dataset.add_frame(frame, task=task)
        if m == "indict":
            return dataset.add_frame({**frame, "task": task})
        if m == "plain":
            return dataset.add_frame(frame)
        try:
            dataset.add_frame(frame, task=task); _mode["v"] = "kw"; return
        except TypeError:
            pass
        try:
            dataset.add_frame({**frame, "task": task}); _mode["v"] = "indict"; return
        except TypeError:
            pass
        dataset.add_frame(frame); _mode["v"] = "plain"

    def save_ep(task):
        try:
            dataset.save_episode()
        except TypeError:
            dataset.save_episode(task=task)

    def to_img(a):
        a = np.asarray(a)
        if a.ndim == 3 and a.shape[2] >= 3:
            a = a[:, :, :3]
        if a.dtype != np.uint8:
            a = np.clip(a, 0, 255).astype(np.uint8)
        return np.ascontiguousarray(a)

    total_src = total_kept = 0
    for i, f in enumerate(files):
        d = np.load(f)
        obs = np.asarray(d["obs"], dtype=np.float32)
        act = np.asarray(d["actions"], dtype=np.float32)
        iw, io = d["images_wrist"], d["images_over"]
        T = min(len(obs), len(act), len(iw), len(io))

        idx = list(range(0, T, s))
        # 마지막 프레임(물체를 놓은 최종 상태)은 항상 남긴다
        if idx[-1] != T - 1:
            idx.append(T - 1)

        for t in idx:
            add_frame({
                "observation.images.wrist": to_img(iw[t]),
                "observation.images.over": to_img(io[t]),
                "observation.state": obs[t][:PROPRIO_DIM],
                "action": act[t],
            }, TASK)
        save_ep(TASK)

        total_src += T
        total_kept += len(idx)
        if i % 20 == 0 or i == len(files) - 1:
            print(f"[convert]  [{i+1}/{len(files)}] {os.path.basename(f)}: "
                  f"{T} -> {len(idx)} frames  (누적 {total_kept})")

    if hasattr(dataset, "finalize"):
        dataset.finalize()
    elif hasattr(dataset, "consolidate"):
        dataset.consolidate()

    print(f"\n[convert] 완료: {len(files)} 에피소드 | "
          f"{total_src} -> {total_kept} 프레임 ({total_kept/total_src*100:.1f}%)")

    if args.push:
        print(f"[convert] Hub 업로드 중: {args.repo_id}")
        dataset.push_to_hub(args.repo_id)
        print("[convert] 업로드 완료")
    else:
        print("[convert] --push 를 주지 않아 로컬에만 저장했습니다")

    # 학습 명령 안내
    epochs_ref = 40
    steps_hint = max(10000, int(round(epochs_ref * total_kept / args.batch_hint / 10000.0) * 10000))
    print()
    print(f"[다음] 약 {epochs_ref}에폭에 해당하는 학습 명령:")
    print(f"  lerobot-train --policy.type=act --dataset.repo_id={args.repo_id} \\")
    print(f"    --steps={steps_hint} --batch_size={args.batch_hint} "
          f"--policy.optimizer_lr=3e-5 \\")
    print(f"    --save_freq={max(10000, steps_hint//4)} "
          f"--output_dir=/data/jinju/act_pickplace_s{s}")
    print(f"[배포] 추론 시 액션 하나를 {s}스텝 유지할 것 "
          f"(정책 {out_fps}Hz vs 물리 {args.src_fps}Hz)")


def main():
    p = argparse.ArgumentParser(description="npz 에피소드를 솎아서 LeRobotDataset으로 변환")
    p.add_argument("--src", default="/data/jinju/bc_data_v5", help="episode_*.npz 폴더")
    p.add_argument("--stride", type=int, default=TRAIN_STRIDE,
                   help=f"N프레임마다 1개 사용 (기본 {TRAIN_STRIDE}, scene_config.TRAIN_STRIDE)")
    p.add_argument("--repo-id", default="jamongsteak/pickplace_vision_v10_s3")
    p.add_argument("--src-fps", type=int, default=RECORD_HZ,
                   help=f"원본 기록 주기 (기본 {RECORD_HZ}, scene_config.RECORD_HZ)")
    p.add_argument("--analyze", action="store_true", help="변환하지 않고 stride 후보만 분석")
    p.add_argument("--sample-n", type=int, default=20, help="분석에 쓸 에피소드 수")
    p.add_argument("--chunk-size", type=int, default=100, help="ACT chunk_size (지평 계산용)")
    p.add_argument("--batch-hint", type=int, default=32, help="학습 명령 안내에 쓸 batch_size")
    p.add_argument("--no-videos", dest="videos", action="store_false",
                   help="mp4 인코딩 대신 png로 저장")
    p.add_argument("--push", action="store_true", help="변환 후 Hub에 업로드")
    p.add_argument("--limit", type=int, default=0, help="앞에서 N개 에피소드만 변환(테스트용)")
    args = p.parse_args()

    files = find_episodes(args.src)

    if args.analyze:
        analyze(files, args.src_fps, args.chunk_size, args.sample_n)
        return

    if args.limit:
        files = files[:args.limit]
        print(f"[convert] --limit {args.limit}: 앞의 {len(files)}개만 변환합니다")
    convert(files, args)


if __name__ == "__main__":
    main()
