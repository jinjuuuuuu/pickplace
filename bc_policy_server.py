#!/usr/bin/env python3
# bc_policy_server.py
# ---------------------------------------------------------------------------
# bc_train_vision.py로 학습한 SmallCNN BC 정책을 act_policy_server.py와 "똑같은"
# 소켓 프로토콜로 서비스한다. 그래서 평가 클라이언트(eval_act_v5_client.py)를
# 한 줄도 고치지 않고 그대로 쓴다.
#
# 왜 이렇게 하나: 성공률 숫자를 ACT v11(100%)과 비교하려면 평가 조건이 완전히
# 같아야 한다. 클라이언트에는 이미 검증된 것들이 들어 있다 — RENDER=True(검은
# 이미지 방지), blank_camera_frames 카운터, 그리퍼 이진화, scene_config 기반
# 평가 격자, ACTION_REPEAT=3. 정책만 갈아끼우는 게 유일하게 공정한 비교다.
#
# 실행 (torch만 있으면 된다. 시스템 파이썬으로 충분):
#   set MODEL_PATH=C:\...\bc_runs\v11_s3_smallcnn\checkpoints\best.pt
#   python bc_policy_server.py
# 다른 창에서:
#   C:\isaacsim\python.bat -u eval_act_v5_client.py
#
# 환경변수:
#   MODEL_PATH     체크포인트 .pt (기본 bc_runs/v11_s3_smallcnn/checkpoints/best.pt)
#   REPLAN_EVERY   청크 중 몇 스텝을 쓰고 다시 관측할지 (기본 20 = 1초 @20Hz).
#                  CHUNK_H로 주면 원본 bc_deploy_vision.py처럼 완전 개루프.
#   POLICY_BIND / POLICY_PORT   act_policy_server.py와 동일
# ---------------------------------------------------------------------------
import collections
import os
import pickle
import socket
import struct

import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms.functional as TF

HERE = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.environ.get(
    "MODEL_PATH", os.path.join(HERE, "bc_runs", "v11_s3_smallcnn", "checkpoints", "best.pt"))
HOST = os.environ.get("POLICY_BIND", "127.0.0.1")
PORT = int(os.environ.get("POLICY_PORT", "5555"))


# ---- 신경망 (bc_train_vision.py와 동일해야 한다) ----------------------------
class SpatialSoftmax(nn.Module):
    def forward(self, x):
        B, C, H, W = x.shape
        a = torch.softmax(x.reshape(B, C, H * W), dim=-1)
        ys, xs = torch.meshgrid(torch.linspace(-1, 1, H, device=x.device),
                                torch.linspace(-1, 1, W, device=x.device), indexing="ij")
        ex = (a * xs.reshape(-1)).sum(-1)
        ey = (a * ys.reshape(-1)).sum(-1)
        return torch.cat([ex, ey], dim=1)


class SmallCNN(nn.Module):
    def __init__(self, out_dim=128):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3, 32, 5, stride=2, padding=2), nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(64, 32, 3, stride=1, padding=1), nn.ReLU(),
        )
        self.sm = SpatialSoftmax()
        self.fc = nn.Linear(64, out_dim)

    def forward(self, x):
        return torch.relu(self.fc(self.sm(self.conv(x))))


class VisionBCPolicy(nn.Module):
    def __init__(self, proprio_dim, action_dim, feat=128):
        super().__init__()
        self.cnn_wrist = SmallCNN(feat)
        self.cnn_over = SmallCNN(feat)
        self.head = nn.Sequential(
            nn.Linear(feat * 2 + proprio_dim, 512), nn.ReLU(),
            nn.Linear(512, 512), nn.ReLU(),
            nn.Linear(512, 256), nn.ReLU(),
            nn.Linear(256, action_dim),
        )

    def forward(self, wrist_img, over_img, proprio):
        return self.head(torch.cat([self.cnn_wrist(wrist_img),
                                    self.cnn_over(over_img), proprio], dim=1))


# ---- 소켓 (act_policy_server.py와 동일 프로토콜) ----------------------------
def send_msg(sock, obj):
    data = pickle.dumps(obj, protocol=4)
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
    return None if data is None else pickle.loads(data)


def decode_img(payload, msg):
    enc = msg.get("enc", "raw")
    if enc == "png":
        import io
        from PIL import Image
        return np.array(Image.open(io.BytesIO(payload)).convert("RGB"), dtype=np.uint8)
    if enc == "zlib":
        import zlib
        payload = zlib.decompress(payload)
    return np.frombuffer(payload, np.uint8).reshape(tuple(msg["shape"])).copy()


# ---- 로드 ------------------------------------------------------------------
if not os.path.isfile(MODEL_PATH):
    raise SystemExit(f"[bc_server] 체크포인트가 없습니다: {MODEL_PATH}\n"
                     f"  MODEL_PATH=... python bc_policy_server.py")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ckpt = torch.load(MODEL_PATH, map_location=device, weights_only=False)

CHUNK_H = int(ckpt["chunk_h"])
JOINT_DIM = int(ckpt["joint_dim"])
PROPRIO_DIM = int(ckpt["proprio_dim"])
IMG_SIZE = int(ckpt["img_size"])
FPS = int(ckpt.get("fps", 20))
REPLAN_EVERY = int(os.environ.get("REPLAN_EVERY", "20"))
REPLAN_EVERY = max(1, min(REPLAN_EVERY, CHUNK_H))

policy = VisionBCPolicy(PROPRIO_DIM, CHUNK_H * JOINT_DIM).to(device)
policy.load_state_dict(ckpt["policy"])
policy.eval()

ns = ckpt.get("norm_stats")
if ns is None:
    ns = torch.load(os.path.join(os.path.dirname(MODEL_PATH), "..", "norm_stats.pt"),
                    map_location="cpu", weights_only=False)
pro_mean = np.asarray(ns["proprio_mean"], dtype=np.float32)
pro_std = np.asarray(ns["proprio_std"], dtype=np.float32)
act_mean = np.asarray(ns["act_mean"], dtype=np.float32)
act_std = np.asarray(ns["act_std"], dtype=np.float32)

print(f"[bc_server] {MODEL_PATH}")
print(f"[bc_server] epoch={ckpt.get('epoch')} val_loss={ckpt.get('val_loss'):.6f} "
      f"| chunk_h={CHUNK_H} ({CHUNK_H/FPS:.1f}s @{FPS}Hz) joint_dim={JOINT_DIM} "
      f"proprio_dim={PROPRIO_DIM} img={IMG_SIZE}")
print(f"[bc_server] REPLAN_EVERY={REPLAN_EVERY} "
      f"({REPLAN_EVERY/FPS:.2f}초마다 재관측; {CHUNK_H}로 주면 완전 개루프) | device={device}")
print(f"[bc_server] 그리퍼 이진화는 클라이언트(binarize_gripper)가 한다 — 여기선 원본을 보낸다")

last_imgs = {"wrist": None, "over": None}
queue = collections.deque()


def prep(img):
    """prepare_bc_data.resize_uint8과 같은 경로 — 학습·추론 전처리가 어긋나면 조용히 실패한다."""
    t = torch.from_numpy(np.ascontiguousarray(img)).permute(2, 0, 1)      # HWC uint8 -> CHW
    t = TF.resize(t, [IMG_SIZE, IMG_SIZE], antialias=True)
    return t.unsqueeze(0).to(device).float().div_(255.0)


def predict(state):
    pro = (np.asarray(state, dtype=np.float32)[:PROPRIO_DIM] - pro_mean) / pro_std
    with torch.no_grad():
        out = policy(prep(last_imgs["wrist"]), prep(last_imgs["over"]),
                     torch.from_numpy(pro).unsqueeze(0).to(device))
    chunk = out.squeeze(0).cpu().numpy().reshape(CHUNK_H, JOINT_DIM) * act_std + act_mean
    return chunk[:REPLAN_EVERY]


srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
srv.bind((HOST, PORT))
srv.listen(1)
print(f"[bc_server] listening on {HOST}:{PORT} - 이제 Isaac Sim 클라이언트를 실행하세요.")

try:
    while True:
        conn, addr = srv.accept()
        print(f"[bc_server] client connected: {addr}")
        with conn:
            while True:
                msg = recv_msg(conn)
                if msg is None:
                    print("[bc_server] client disconnected")
                    break
                cmd = msg.get("cmd")
                if cmd == "reset":
                    queue.clear()
                    last_imgs["wrist"] = last_imgs["over"] = None
                    send_msg(conn, {"ok": True})
                elif cmd == "act":
                    if "wrist" in msg:
                        last_imgs["wrist"] = decode_img(msg["wrist"], msg)
                        last_imgs["over"] = decode_img(msg["over"], msg)
                    elif last_imgs["wrist"] is None:
                        send_msg(conn, {"error": "첫 호출에는 이미지가 있어야 합니다"})
                        continue
                    if not queue:
                        queue.extend(predict(msg["state"]))
                    action = queue.popleft()
                    # need_obs=True면 클라이언트가 다음 호출에 이미지를 실어 보낸다.
                    send_msg(conn, {"action": [float(x) for x in action],
                                    "need_obs": len(queue) == 0})
                elif cmd == "bye":
                    send_msg(conn, {"ok": True})
                    print("[bc_server] client said bye")
                    break
                else:
                    send_msg(conn, {"error": f"unknown cmd {cmd}"})
        print("[bc_server] 다음 클라이언트 대기 중... (종료하려면 Ctrl+C)")
except KeyboardInterrupt:
    print("\n[bc_server] stopped")
finally:
    srv.close()
