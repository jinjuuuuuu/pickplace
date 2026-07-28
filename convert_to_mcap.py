#!/usr/bin/env python3
# convert_to_mcap.py  — 우리 수집 npz 에피소드 → Foxglove용 mcap
# ---------------------------------------------------------------------------
# WSL2에서 실행 (파일이 온전히 읽히는 곳):
#   pip install mcap pillow numpy      # 최초 1회
#   python /mnt/c/Users/user/Desktop/claude_jetbot/convert_to_mcap.py            # episode_0000 변환
#   python .../convert_to_mcap.py /mnt/c/Users/user/Desktop/claude_jetbot/bc_data_v3/episode_0005.npz
#
# 만드는 토픽 (Foxglove Studio에서 바로 열림):
#   /camera/overhead  (CompressedImage, jpeg)
#   /camera/wrist     (CompressedImage, jpeg)
#   /state            (JSON: 관절 q0..q8, 큐브 cube_x/y/z)  ← Plot 패널로 그래프
#   /action           (JSON: 목표 a0..a8)
# ---------------------------------------------------------------------------
import sys, os, io, json, base64
import numpy as np
from PIL import Image
from mcap.writer import Writer

FPS = 30
DEFAULT = "/mnt/c/Users/user/Desktop/claude_jetbot/bc_data_v3/episode_0000.npz"

def jpeg_b64(rgb):
    im = Image.fromarray(np.ascontiguousarray(rgb[:, :, :3].astype(np.uint8)))
    buf = io.BytesIO(); im.save(buf, format="JPEG", quality=90)
    return base64.b64encode(buf.getvalue()).decode("ascii")

# Foxglove 표준 스키마 (jsonschema 인코딩) — Studio가 이름으로 인식
IMG_SCHEMA = json.dumps({
    "type": "object",
    "properties": {
        "timestamp": {"type": "object", "properties": {
            "sec": {"type": "integer"}, "nsec": {"type": "integer"}}},
        "frame_id": {"type": "string"},
        "data": {"type": "string", "contentEncoding": "base64"},
        "format": {"type": "string"},
    },
}).encode()

def main():
    src = sys.argv[1] if len(sys.argv) > 1 else DEFAULT
    out = os.path.splitext(src)[0] + ".mcap"
    d = np.load(src)
    iw = d["images_wrist"]; io_ = d["images_over"]
    obs = d["obs"]; act = d["actions"]
    T = min(len(iw), len(io_), len(obs), len(act))
    print(f"[mcap] {os.path.basename(src)} | {T} 프레임 → {out}")

    with open(out, "wb") as f:
        w = Writer(f); w.start()
        img_sid = w.register_schema(name="foxglove.CompressedImage",
                                    encoding="jsonschema", data=IMG_SCHEMA)
        st_sid  = w.register_schema(name="jetbot/state",  encoding="jsonschema", data=b'{"type":"object"}')
        ac_sid  = w.register_schema(name="jetbot/action", encoding="jsonschema", data=b'{"type":"object"}')
        ch_over  = w.register_channel(topic="/camera/overhead", message_encoding="json", schema_id=img_sid)
        ch_wrist = w.register_channel(topic="/camera/wrist",    message_encoding="json", schema_id=img_sid)
        ch_state = w.register_channel(topic="/state",  message_encoding="json", schema_id=st_sid)
        ch_act   = w.register_channel(topic="/action", message_encoding="json", schema_id=ac_sid)

        for i in range(T):
            t_ns = int(i * 1e9 / FPS)
            sec, nsec = t_ns // 1_000_000_000, t_ns % 1_000_000_000
            for ch, imgs, fid in [(ch_over, io_, "overhead"), (ch_wrist, iw, "wrist")]:
                msg = {"timestamp": {"sec": sec, "nsec": nsec}, "frame_id": fid,
                       "data": jpeg_b64(imgs[i]), "format": "jpeg"}
                w.add_message(channel_id=ch, log_time=t_ns, publish_time=t_ns,
                              data=json.dumps(msg).encode())
            o = obs[i]
            state = {f"q{j}": float(o[j]) for j in range(9)}
            state.update({"cube_x": float(o[24]), "cube_y": float(o[25]), "cube_z": float(o[26])})
            w.add_message(channel_id=ch_state, log_time=t_ns, publish_time=t_ns,
                          data=json.dumps(state).encode())
            action = {f"a{j}": float(act[i][j]) for j in range(min(9, act.shape[1]))}
            w.add_message(channel_id=ch_act, log_time=t_ns, publish_time=t_ns,
                          data=json.dumps(action).encode())
        w.finish()
    print(f"[mcap] 완료! Foxglove Studio에서 열기: {out}")

if __name__ == "__main__":
    main()
