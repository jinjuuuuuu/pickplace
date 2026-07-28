# ACT (Action Chunking Transformer) — Franka Pick & Place

시각(카메라) 기반 모방학습 파이프라인. 단일스텝 MLP-BC가 큐브에 도달/파지하지 못한
근본 한계(행동이 큐브와 무상관·하강동작이 평균에서 상쇄)를 해결하기 위해,
오픈소스 ACT를 가져와 Isaac Sim에 적용했다.

## 출처 (가져온 오픈소스)

- 원본 ACT: https://github.com/tonyzhaozh/act
- Franka 포팅: https://github.com/manishalingala/ACTfranka

`detr/`, `policy.py`, `act_dataset.py`(원본 `training/utils.py`)는 위 레포의 **모델·학습
코드를 그대로 가져온 것**이다(디버그용 IPython 호출만 제거). Isaac Sim 연동을 위한
데이터 수집/배포 스크립트(`act_collect_isaac.py`, `act_deploy_isaac.py`)와 설정
(`config.py`), 학습 진입점(`act_train.py`)만 이 환경에 맞춰 새로 작성했다.

## 폴더 구조

```
ACT/
├─ config.py               # 모든 경로/카메라/하이퍼파라미터 (여기만 수정)
├─ act_collect_isaac.py    # [Isaac] 전문가 데모 수집 -> .npz (카메라+관절+행동)
├─ act_train.py            # [일반 Python+GPU] ACT 학습
├─ act_deploy_isaac.py     # [Isaac] 학습된 정책으로 카메라 기반 제어
├─ policy.py               # (가져옴) ACTPolicy / CNNMLPPolicy
├─ act_dataset.py          # (가져옴) 데이터로더 + 정규화 + 헬퍼 (.npz 읽도록 I/O만 수정)
├─ detr/                   # (가져옴) DETR 기반 CVAE 백본/트랜스포머
├─ requirements_act.txt
├─ data/                   # (자동 생성) episode_*.npz
└─ checkpoints/            # (자동 생성) policy_best.ckpt, dataset_stats.pkl
```

## 카메라 배치 (ACT는 시각 모델이라 필수)

`config.py`의 `CAMERA_NAMES = ["top", "wrist"]`

- **wrist**: 손(`panda_hand`)에 부착되어 함께 움직임 → 정밀 정렬·파지에 핵심
- **top**: 천장 고정(`OVERHEAD_CAM_POS`) → 큐브/타겟 위치 파악

수집·배포 스크립트가 동일하게 이 두 카메라를 생성한다. 관측은
`[관절1..7, 그리퍼(0/1)]`(qpos, 8차원) + 카메라 2장 RGB(640×480)이다.

## 설치

```bash
# 1) 학습용(GPU 있는 일반 파이썬 환경 권장)
pip install -r requirements_act.txt

# 2) 수집/배포용 (Isaac Sim 파이썬에 설치)
"C:\isaacsim\python.bat" -m pip install einops opencv-python
```
(torch/torchvision/numpy는 Isaac Sim에 이미 포함. 학습 환경엔 CUDA 버전 torch 설치.)
※ h5py는 쓰지 않는다 — Windows Isaac 파이썬에서 h5py가 DLL 로드에 실패하는
  문제가 있어, 데이터는 numpy(.npz)로 저장/로드하도록 했다.

## 전체 과정

### 1단계 — 데이터 수집 (Isaac Sim)
```bat
"C:\isaacsim\python.bat" "C:\Users\user\Desktop\claude_jetbot\ACT\act_collect_isaac.py"
```
- `PickPlaceController`(전문가)로 성공 데모를 생성, `data/episode_*.npz` 저장.
- 모든 에피소드는 `EPISODE_LEN`(기본 200) 프레임으로 균일 리샘플링(ACT는 길이 통일 필요).
- 기본 50개 성공 에피소드 수집(`NUM_EPISODES`). 더 많을수록 좋다(100+ 권장).

### 2단계 — 학습 (GPU)
```bash
python act_train.py
```
- `checkpoints/policy_best.ckpt`, `dataset_stats.pkl` 생성.
- l1 loss가 충분히 내려가야(대략 <0.1) 잘 학습된 것. GPU에서 수십 분~수 시간.

### 3단계 — 배포 (Isaac Sim)
```bat
"C:\isaacsim\python.bat" "C:\Users\user\Desktop\claude_jetbot\ACT\act_deploy_isaac.py"
```
- 카메라 영상으로 매 스텝 미래 행동 chunk를 예측, temporal aggregation으로 부드럽게 실행.
- `EVAL_CUBE_POS`/`EVAL_TARGET_POS`로 배포 시 큐브/타겟 위치를 바꿔 일반화 확인.

## 주요 설정 (config.py)

| 항목 | 기본 | 설명 |
|---|---|---|
| `CAMERA_NAMES` | top, wrist | 사용 카메라 |
| `EPISODE_LEN` | 200 | 에피소드 리샘플 길이 |
| `NUM_EPISODES` | 50 | 수집할 성공 데모 수 |
| `CHUNK_SIZE` | 100 | 한 번에 예측하는 미래 스텝 수(num_queries) |
| `KL_WEIGHT` | 10 | CVAE KL 가중치 |
| `NUM_EPOCHS` | 3000 | 학습 epoch |
| `TEMPORAL_AGG` | True | 배포 시 시간축 앙상블 |

## 알아둘 점 / 트러블슈팅

- **GPU 필수(학습)**: ResNet18 백본+트랜스포머라 CPU 학습은 비현실적.
- **ResNet18 가중치**: 최초 실행 시 torchvision이 사전학습 가중치를 다운로드(인터넷 필요).
- **카메라 API**: `isaacsim.sensors.camera.Camera` 기준. Isaac 5.1 빌드에 따라
  카메라 프림 회전(`SetRotate`)이나 `get_rgba()` 반환형이 다를 수 있다. 수집 직후
  `data/episode_0.npz`의 이미지를 한 장 열어 **장면이 제대로 찍혔는지 꼭 확인**할 것
  (예: `python -c "import numpy as np,matplotlib.pyplot as plt; d=np.load(r'ACT/data/episode_0.npz'); plt.imshow(d['image_top'][100]); plt.show()"`)
  (검은 화면이면 카메라 방향/초기화/렌더 워밍업을 조정).
- **데이터 양**: 50개로 시작하되, 일반화가 약하면 100~200개로 늘리는 게 가장 효과적.
- **이전 MLP 방식**(`../bc_*.py`)은 이 ACT 파이프라인으로 대체됨. ACT 폴더는 독립적으로 동작.
```
