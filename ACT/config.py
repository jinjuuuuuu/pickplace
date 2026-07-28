# config.py  -  ACT (Action Chunking Transformer) settings
# Original ACT: https://github.com/tonyzhaozh/act
# Franka port : https://github.com/manishalingala/ACTfranka  (vendored here)
# 데이터/학습/배포 스크립트가 모두 이 파일을 읽는다.
import os
import torch

device = "cuda" if torch.cuda.is_available() else "cpu"
os.environ["DEVICE"] = device

# === Paths ===
ACT_DIR        = r"C:\Users\user\Desktop\claude_jetbot\ACT"
DATASET_DIR    = r"C:\Users\user\Desktop\claude_jetbot\ACT\data"
CHECKPOINT_DIR = r"C:\Users\user\Desktop\claude_jetbot\ACT\checkpoints"
FRANKA_USD     = r"C:\Users\user\Desktop\claude_jetbot\franka\franka.usd"

# === Cameras (top + front, both FIXED) ===
# ACTfranka 원본도 top+front 고정 카메라 사용. 손목(articulation 자식) 카메라는
# render product가 불안정해 standalone 크래시 위험 -> 고정 카메라로 안정화.
# 첫 수집 후 저장된 이미지를 보고 위치/각도(euler deg)를 미세조정할 것.
CAMERA_NAMES = ["top", "front"]
CAM_WIDTH    = 320
CAM_HEIGHT   = 240
OVERHEAD_CAM_POS   = [0.45, 0.0, 1.4]
OVERHEAD_CAM_EULER = [0.0, 90.0, 0.0]
FRONT_CAM_POS      = [1.25, 0.0, 0.55]
FRONT_CAM_EULER    = [0.0, 20.0, 180.0]

# === Dims (detr 모델 state_dim=8 하드코딩) : [joint1..7, gripper(0/1)] ===
STATE_DIM    = 8
ACTION_DIM   = 8

# === Episode ===
EPISODE_LEN  = 200
NUM_EPISODES = 50
COLLECT_MAX_STEPS = 1500
FINGER_OPEN  = 0.04
FINGER_CLOSE = 0.0
GRASP_FINGER_THRESH = 0.03

CUBE_X_RANGE   = (0.35, 0.55)
CUBE_Y_RANGE   = (-0.20, 0.20)
CUBE_Z         = 0.025
TARGET_X_RANGE = (0.35, 0.55)
TARGET_Y_RANGE = (-0.20, 0.20)
TARGET_Z       = 0.025
MIN_DISTANCE   = 0.15

# === ACT policy hyperparams ===
CHUNK_SIZE   = 100
KL_WEIGHT    = 10
HIDDEN_DIM   = 512
DIM_FEEDFORWARD = 3200
BACKBONE     = "resnet18"
ENC_LAYERS   = 4
DEC_LAYERS   = 7
NHEADS       = 8

# === Training ===
LR           = 1e-5
LR_BACKBONE  = 1e-5
WEIGHT_DECAY = 1e-4
BATCH_SIZE   = 8
NUM_EPOCHS   = 3000
SEED         = 42

# === Deploy ===
TEMPORAL_AGG     = True
DEPLOY_MAX_STEPS = 600
EVAL_CUBE_POS    = [0.45, 0.10, 0.025]
EVAL_TARGET_POS  = [0.50, -0.15, 0.025]
START_POSE = [0.0, -0.3, 0.0, -2.5, 0.0, 2.2, 0.8]


def build_policy_config():
    # detr model build args (camera_names 등 override)
    return {
        "lr": LR,
        "lr_backbone": LR_BACKBONE,
        "weight_decay": WEIGHT_DECAY,
        "device": device,
        "num_queries": CHUNK_SIZE,
        "kl_weight": KL_WEIGHT,
        "hidden_dim": HIDDEN_DIM,
        "dim_feedforward": DIM_FEEDFORWARD,
        "backbone": BACKBONE,
        "enc_layers": ENC_LAYERS,
        "dec_layers": DEC_LAYERS,
        "nheads": NHEADS,
        "camera_names": CAMERA_NAMES,
        "policy_class": "ACT",
    }
