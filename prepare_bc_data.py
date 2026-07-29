#!/usr/bin/env python3
# prepare_bc_data.py
# ---------------------------------------------------------------------------
# LeRobot v3.0 데이터셋(허브)을 BC 학습용 npy 캐시로 굽는다.
#
# 왜 필요한가: v11_s3는 codebase_version v3.0 (카메라당 mp4 한 개에 전 에피소드를
# 이어붙인 형식)이다. 데스크톱 파이썬의 lerobot은 0.4.4라 v3.0을 못 읽고,
# Isaac Sim 파이썬(3.11)에도 lerobot 0.4.4가 들어 있다. 게다가 LeRobotDataset은
# 프레임마다 mp4를 랜덤 디코딩하므로 100에폭 학습에는 너무 느리다.
#
# 대신 여기서 한 번만 순차 디코딩해서 84x84 uint8 배열로 저장한다.
# v3.0은 에피소드를 순서대로 이어붙이므로 (meta/episodes의 dataset_from_index와
# from_timestamp가 둘 다 단조 증가) 순차 디코딩 순서 == 데이터 행 순서다.
# 이 스크립트가 그 가정을 프레임 수로 검증한다.
#
# 실행 (시스템 파이썬 3.12 — pyav가 여기 있다):
#   python prepare_bc_data.py --repo-id jamongsteak/pickplace_vision_v11_s3
#
# 사내 프록시(Somansa) 때문에 python SSL이 막히면:
#   $env:REQUESTS_CA_BUNDLE="$env:USERPROFILE\.certs\win-ca-bundle.pem"
# ---------------------------------------------------------------------------
import argparse
import json
import os

import numpy as np


def resize_uint8(frame_hwc, size):
    """학습·추론이 같은 코드로 리사이즈해야 한다. bc_policy_server.py도 이걸 쓴다."""
    import torch
    import torchvision.transforms.functional as TF

    t = torch.from_numpy(np.ascontiguousarray(frame_hwc)).permute(2, 0, 1)  # HWC->CHW uint8
    t = TF.resize(t, [size, size], antialias=True)
    return t.permute(1, 2, 0).numpy()


def decode_video(path, n_expect, size, out_path):
    import av
    from tqdm import tqdm

    container = av.open(path)
    stream = container.streams.video[0]
    n_stream = stream.frames
    if n_stream != n_expect:
        raise SystemExit(
            f"[prepare] 프레임 수 불일치: {os.path.basename(path)} = {n_stream}, "
            f"parquet 행 = {n_expect}. 순차 디코딩 정렬 가정이 깨졌다.")

    arr = np.lib.format.open_memmap(out_path, mode="w+", dtype=np.uint8,
                                    shape=(n_expect, size, size, 3))
    i = 0
    for frame in tqdm(container.decode(video=0), total=n_expect, desc=os.path.basename(out_path)):
        arr[i] = resize_uint8(frame.to_ndarray(format="rgb24"), size)
        i += 1
    container.close()
    if i != n_expect:
        raise SystemExit(f"[prepare] 디코딩된 프레임 {i} != 기대 {n_expect}")
    arr.flush()
    return arr


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo-id", default="jamongsteak/pickplace_vision_v11_s3")
    ap.add_argument("--img-size", type=int, default=84)
    ap.add_argument("--out", default=None, help="기본: ./bc_cache_<데이터셋이름>_<크기>")
    ap.add_argument("--local-dir", default=None, help="이미 받아둔 스냅샷 폴더가 있으면 지정")
    args = ap.parse_args()

    if args.local_dir:
        root = args.local_dir
    else:
        from huggingface_hub import snapshot_download
        root = snapshot_download(args.repo_id, repo_type="dataset")
    print(f"[prepare] 데이터셋 경로: {root}")

    info = json.load(open(os.path.join(root, "meta", "info.json"), encoding="utf-8"))
    fps = int(info["fps"])
    n_frames = int(info["total_frames"])
    n_eps = int(info["total_episodes"])
    state_dim = int(info["features"]["observation.state"]["shape"][0])
    act_dim = int(info["features"]["action"]["shape"][0])
    print(f"[prepare] {n_eps} 에피소드 | {n_frames} 프레임 | {fps}Hz | "
          f"state={state_dim} action={act_dim} | v{info['codebase_version']}")

    import pandas as pd
    data_files = sorted(
        os.path.join(dp, f)
        for dp, _, fs in os.walk(os.path.join(root, "data")) for f in fs if f.endswith(".parquet"))
    df = pd.concat([pd.read_parquet(f) for f in data_files], ignore_index=True)
    df = df.sort_values("index").reset_index(drop=True)
    if len(df) != n_frames:
        raise SystemExit(f"[prepare] parquet 행 {len(df)} != info.json {n_frames}")

    states = np.stack(df["observation.state"].to_numpy()).astype(np.float32)
    actions = np.stack(df["action"].to_numpy()).astype(np.float32)
    ep_index = df["episode_index"].to_numpy().astype(np.int64)
    if not np.all(np.diff(ep_index) >= 0):
        raise SystemExit("[prepare] episode_index가 단조 증가가 아니다 — 순차 정렬 가정이 깨졌다.")

    out = args.out or os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        f"bc_cache_{args.repo_id.split('/')[-1]}_{args.img_size}")
    os.makedirs(out, exist_ok=True)

    cams = {}
    for key, short in (("observation.images.wrist", "wrist"), ("observation.images.over", "over")):
        vids = sorted(
            os.path.join(dp, f)
            for dp, _, fs in os.walk(os.path.join(root, "videos", key)) for f in fs
            if f.endswith(".mp4"))
        if len(vids) != 1:
            raise SystemExit(f"[prepare] {key}: mp4가 {len(vids)}개다. 이 스크립트는 "
                             f"단일 파일(chunk 하나)만 가정한다.")
        cams[short] = decode_video(vids[0], n_frames, args.img_size,
                                   os.path.join(out, f"{short}_{args.img_size}.npy"))

    np.savez(os.path.join(out, "meta.npz"),
             states=states, actions=actions, episode_index=ep_index,
             fps=np.int64(fps), img_size=np.int64(args.img_size),
             repo_id=np.array(args.repo_id))

    # --- 검증 출력: 검은 프레임/가려짐을 학습 전에 잡는다 (프로젝트 진단 순서 1번) ---
    print(f"\n[prepare] 저장 완료 → {out}")
    for short, arr in cams.items():
        sample = arr[:: max(1, n_frames // 2000)].astype(np.float32)
        per_frame_std = arr[:: max(1, n_frames // 2000)].reshape(-1, args.img_size ** 2 * 3).std(1)
        n_flat = int((per_frame_std < 1.0).sum())
        print(f"  {short:5s} {arr.shape} | mean {sample.mean():6.1f} std {sample.std():5.1f} "
              f"| 평탄(std<1) 프레임 {n_flat}/{len(per_frame_std)}")
        if n_flat:
            print(f"    ⚠ {short}에 단색 프레임이 있다. 검은 이미지로 학습되지 않는지 확인할 것.")
    print(f"  state  {states.shape} | action {actions.shape}")
    print(f"  gripper 라벨 고유값: {np.unique(np.round(actions[:, 7:9], 4))[:8]}")
    print(f"  에피소드 길이: {np.bincount(ep_index).min()}~{np.bincount(ep_index).max()} 프레임")


if __name__ == "__main__":
    main()
