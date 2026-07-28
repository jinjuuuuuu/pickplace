#!/usr/bin/env python3
# act_policy_server.py
# ---------------------------------------------------------------------------
# conda `lerobot` 환경(Python 3.12)에서 실행. v5 ACT 모델을 로드하고,
# 소켓으로 관측(이미지+상태)을 받아 액션(9)을 돌려준다.
# Isaac Sim 클라이언트(eval_act_v5_client.py)와 짝을 이룬다.
#
# 왜 분리하나: v5를 학습시킨 lerobot fork(0.6.1)는 Python 3.12/numpy2 필요라
# Isaac Sim(3.11/numpy1.26)과 한 프로세스에 못 올린다. 그래서 정책은 이쪽,
# 시뮬은 저쪽에서 각자 자기 환경으로 돌리고 소켓으로 대화한다.
#
# 실행:
#   conda activate lerobot
#   python ~/pickplace/act_policy_server.py
# ---------------------------------------------------------------------------
import os
import socket
import struct
import pickle
import numpy as np

# * lerobot-train --output_dir 의 체크포인트. ls 로 실제 경로 확인 후 맞출 것.
# 코드를 고치지 않고 환경변수로도 지정 가능:
#   MODEL_PATH=/data/jinju/act_pickplace_s3/checkpoints/100000/pretrained_model python act_policy_server.py
MODEL_PATH = os.environ.get(
    "MODEL_PATH", "/data/jinju/act_pickplace_v5/checkpoints/last/pretrained_model")
TASK = "pick up the cube and place it on the target"
# 기본은 같은 머신만 접속 가능(127.0.0.1). 시뮬을 다른 PC에서 돌리려면
# POLICY_BIND=0.0.0.0 으로 띄우고, 클라이언트에 POLICY_HOST=<이 서버 IP>를 준다.
#   POLICY_BIND=0.0.0.0 MODEL_PATH=... python act_policy_server.py
# 주의: 이미지가 매 스텝 오가므로(320x240 x2 = 약 460KB) LAN이 아니면 느리다.
HOST = os.environ.get("POLICY_BIND", "127.0.0.1")
PORT = int(os.environ.get("POLICY_PORT", "5555"))

# --- 폐루프 옵션 (둘 다 추론 시점 설정이라 재학습이 필요 없다) ----------------
# 학습된 ACT는 chunk_size=100, n_action_steps=100이다. 즉 한 번 관측하고 100
# 정책스텝을 개루프로 실행한다 - ACTION_REPEAT=3이면 300 시뮬스텝(5초)을 눈을
# 감고 움직이는 셈이라, 파지 직전 마지막 몇 cm를 보정할 수 없다.
#
#   N_ACTION_STEPS=10        chunk 100개 중 10개만 쓰고 다시 관측 (0.5초마다 보정)
#   TEMPORAL_ENSEMBLE=0.01   ACT 논문의 temporal aggregation. 매 스텝 다시 관측하고
#                            겹치는 chunk들을 지수가중 평균한다 (n_action_steps=1 강제)
#
# 예:  TEMPORAL_ENSEMBLE=0.01 MODEL_PATH=... python act_policy_server.py
# 둘 다 안 주면 학습 시 설정(개루프 100스텝) 그대로 돈다.
_n_steps = os.environ.get("N_ACTION_STEPS", "").strip()
N_ACTION_STEPS = int(_n_steps) if _n_steps else None
_te = os.environ.get("TEMPORAL_ENSEMBLE", "").strip()
TEMPORAL_ENSEMBLE = float(_te) if _te else None

import torch
from lerobot.policies.act.modeling_act import ACTPolicy
from lerobot.policies.factory import make_pre_post_processors

# predict_action / get_safe_torch_device 의 모듈 위치가 lerobot 배포판마다 다르다.
# 워크스테이션의 /data/lerobot(0.6.1 fork)은 common/, 공식 0.6.1은 utils/ 쪽이다.
try:
    from lerobot.common.control_utils import predict_action
except ImportError:
    from lerobot.utils.control_utils import predict_action
# get_safe_torch_device는 안 쓴다. fork마다 위치가 다른데, 하는 일이 "문자열을
# torch.device로 바꾸고 cuda가 없으면 cpu로 내린다"뿐이고 그 판단은 아래에서
# torch.cuda.is_available()로 이미 하고 있다.


# ---- 이미지 디코딩 / 관측 필요 여부 ----------------------------------------
# 클라이언트는 정책이 새 관측을 실제로 쓰는 스텝에만 이미지를 보낸다. ACT는
# n_action_steps개를 큐에 채워두고 하나씩 꺼내므로, 큐가 비어 있지 않으면 이미지가
# 쓰이지 않는다. 그 스텝에는 직전 프레임을 재사용한다(예측에 영향 없음).
last_imgs = {"wrist": None, "over": None}


def decode_img(payload, msg):
    enc = msg.get("enc", "raw")
    if enc == "png":
        from PIL import Image
        import io
        return np.array(Image.open(io.BytesIO(payload)).convert("RGB"), dtype=np.uint8)
    if enc == "zlib":
        import zlib
        payload = zlib.decompress(payload)
    return np.frombuffer(payload, np.uint8).reshape(tuple(msg["shape"])).copy()


def needs_observation():
    """다음 호출에서 정책이 새 관측을 쓰는가 (= 액션 큐가 비었는가)."""
    q = getattr(policy, "_action_queue", None)
    if q is None:
        return True          # 내부 구조가 다른 버전이면 안전하게 매번 요청
    return len(q) == 0


def send_msg(sock, obj):
    data = pickle.dumps(obj, protocol=4)   # protocol 4 = 3.11/3.12 호환
    sock.sendall(struct.pack(">I", len(data)) + data)


def recvall(sock, n):
    buf = bytearray()
    while len(buf) < n:
        chunk = sock.recv(n - len(buf))
        if not chunk:
            return None
        buf.extend(chunk)
    return bytes(buf)


def recv_msg(sock):
    raw = recvall(sock, 4)
    if raw is None:
        return None
    n = struct.unpack(">I", raw)[0]
    data = recvall(sock, n)
    if data is None:
        return None
    return pickle.loads(data)


if not os.path.isdir(MODEL_PATH):
    raise SystemExit(
        f"[server] 체크포인트 폴더가 없습니다: {MODEL_PATH}\n"
        f"  ls /data/jinju/act_pickplace_v5/checkpoints/  로 확인 후 MODEL_PATH 수정.")

print(f"[server] loading policy: {MODEL_PATH}")
policy = ACTPolicy.from_pretrained(MODEL_PATH)
policy.config.device = "cuda" if torch.cuda.is_available() else "cpu"
device = torch.device(policy.config.device)

print(f"[server] 학습 시 설정: chunk_size={policy.config.chunk_size} "
      f"n_action_steps={policy.config.n_action_steps} "
      f"temporal_ensemble_coeff={policy.config.temporal_ensemble_coeff}")

if TEMPORAL_ENSEMBLE is not None:
    # ACTPolicy.__init__은 config를 보고 ensembler를 만든다. from_pretrained 이후에
    # config만 바꿔도 ensembler가 없으므로 직접 붙여준다.
    from lerobot.policies.act.modeling_act import ACTTemporalEnsembler
    policy.config.n_action_steps = 1
    policy.config.temporal_ensemble_coeff = TEMPORAL_ENSEMBLE
    policy.temporal_ensembler = ACTTemporalEnsembler(TEMPORAL_ENSEMBLE, policy.config.chunk_size)
    print(f"[server] → temporal ensembling ON (coeff={TEMPORAL_ENSEMBLE}, "
          f"매 정책스텝마다 재관측)")
elif N_ACTION_STEPS is not None:
    if N_ACTION_STEPS > policy.config.chunk_size:
        raise SystemExit(f"[server] N_ACTION_STEPS({N_ACTION_STEPS})는 "
                         f"chunk_size({policy.config.chunk_size})를 넘을 수 없습니다")
    policy.config.n_action_steps = N_ACTION_STEPS
    print(f"[server] → n_action_steps={N_ACTION_STEPS} "
          f"({N_ACTION_STEPS}스텝마다 재관측)")
else:
    print(f"[server] → 학습 시 설정 그대로 (개루프 "
          f"{policy.config.n_action_steps}스텝). 폐루프로 바꾸려면 "
          f"TEMPORAL_ENSEMBLE=0.01 또는 N_ACTION_STEPS=10")

policy.to(device).eval()
policy.reset()
preprocessor, postprocessor = make_pre_post_processors(policy.config, pretrained_path=MODEL_PATH)
print(f"[server] policy loaded. device={device}")

srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
srv.bind((HOST, PORT))
srv.listen(1)
print(f"[server] listening on {HOST}:{PORT} - 이제 Isaac Sim 클라이언트를 실행하세요.")

try:
    while True:
        conn, addr = srv.accept()
        print(f"[server] client connected: {addr}")
        with conn:
            while True:
                msg = recv_msg(conn)
                if msg is None:
                    print("[server] client disconnected")
                    break
                cmd = msg.get("cmd")
                if cmd == "reset":
                    policy.reset()
                    preprocessor.reset()
                    postprocessor.reset()
                    # 이전 에피소드 프레임이 재사용되지 않도록 캐시도 비운다
                    last_imgs["wrist"] = last_imgs["over"] = None
                    send_msg(conn, {"ok": True})
                elif cmd == "act":
                    if "wrist" in msg:
                        last_imgs["wrist"] = decode_img(msg["wrist"], msg)
                        last_imgs["over"] = decode_img(msg["over"], msg)
                    elif last_imgs["wrist"] is None:
                        send_msg(conn, {"error": "첫 호출에는 이미지가 있어야 합니다"})
                        continue
                    obs = {
                        "observation.images.wrist": last_imgs["wrist"],
                        "observation.images.over":  last_imgs["over"],
                        "observation.state":        np.asarray(msg["state"], dtype=np.float32),
                    }
                    action = predict_action(
                        observation=obs, policy=policy, device=device,
                        preprocessor=preprocessor, postprocessor=postprocessor,
                        use_amp=policy.config.use_amp, task=TASK, robot_type="franka",
                    )
                    act = action.squeeze(0).to("cpu").numpy().astype(float).tolist()
                    send_msg(conn, {"action": act, "need_obs": needs_observation()})
                elif cmd == "bye":
                    send_msg(conn, {"ok": True})
                    print("[server] client said bye")
                    break
                else:
                    send_msg(conn, {"error": f"unknown cmd {cmd}"})
        print("[server] 다음 클라이언트 대기 중... (종료하려면 Ctrl+C)")
except KeyboardInterrupt:
    print("\n[server] stopped")
finally:
    srv.close()
