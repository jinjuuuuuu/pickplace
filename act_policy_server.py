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

# 🔧 lerobot-train --output_dir 의 체크포인트. ls 로 실제 경로 확인 후 맞출 것.
# 코드를 고치지 않고 환경변수로도 지정 가능:
#   MODEL_PATH=/data/jinju/act_pickplace_s3/checkpoints/100000/pretrained_model python act_policy_server.py
MODEL_PATH = os.environ.get(
    "MODEL_PATH", "/data/jinju/act_pickplace_v5/checkpoints/last/pretrained_model")
TASK = "pick up the cube and place it on the target"
HOST = "127.0.0.1"
PORT = 5555

import torch
from lerobot.policies.act.modeling_act import ACTPolicy
from lerobot.policies.factory import make_pre_post_processors
from lerobot.utils.control_utils import predict_action
from lerobot.utils.utils import get_safe_torch_device


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

print(f"[server] loading v5 policy: {MODEL_PATH}")
policy = ACTPolicy.from_pretrained(MODEL_PATH)
policy.config.device = "cuda" if torch.cuda.is_available() else "cpu"
device = get_safe_torch_device(policy.config.device)
policy.to(device).eval()
preprocessor, postprocessor = make_pre_post_processors(policy.config, pretrained_path=MODEL_PATH)
print(f"[server] policy loaded. device={device}")

srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
srv.bind((HOST, PORT))
srv.listen(1)
print(f"[server] listening on {HOST}:{PORT} — 이제 Isaac Sim 클라이언트를 실행하세요.")

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
                    send_msg(conn, {"ok": True})
                elif cmd == "act":
                    shape = tuple(msg["shape"])
                    obs = {
                        "observation.images.wrist": np.frombuffer(msg["wrist"], np.uint8).reshape(shape).copy(),
                        "observation.images.over":  np.frombuffer(msg["over"],  np.uint8).reshape(shape).copy(),
                        "observation.state":        np.asarray(msg["state"], dtype=np.float32),
                    }
                    action = predict_action(
                        observation=obs, policy=policy, device=device,
                        preprocessor=preprocessor, postprocessor=postprocessor,
                        use_amp=policy.config.use_amp, task=TASK, robot_type="franka",
                    )
                    act = action.squeeze(0).to("cpu").numpy().astype(float).tolist()
                    send_msg(conn, {"action": act})
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
