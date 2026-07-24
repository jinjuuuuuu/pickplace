# Franka Pick-and-Place 모방학습 (Isaac Sim + LeRobot ACT)

Franka Panda 로봇팔이 **큐브를 집어 타겟 위치에 내려놓는(pick-and-place)** 동작을,
Isaac Sim에서 수집한 전문가 시연으로 **모방학습(ACT)** 시키는 프로젝트입니다.

```
[Windows] Isaac Sim 5.1                [WSL2 / Linux] conda lerobot            Hugging Face
pick_place_collect.py  ──npz──▶  convert_to_lerobot.py ──push──▶  데이터셋 ──▶ lerobot-train (ACT)
 (전문가 시연 수집)              (LeRobot 포맷 변환)                              (정책 학습)
```

## 이 repo의 파일

| 파일 | 역할 |
|------|------|
| `pick_place_collect.py` | Isaac Sim에서 pick-place 시연을 수집해 `episode_*.npz`로 저장 (관절상태 + 액션 + wrist/overhead 카메라) |
| `convert_to_lerobot.py` | `bc_data_v5/`의 npz들을 `LeRobotDataset`(이미지+상태)으로 변환하고 HF 허브에 업로드 |
| `bc_train_vision_resnet18.py` | (대안 경로) ResNet18 백본 커스텀 BC 학습 스크립트 — **현재 보류**, 아래 주의 참고 |
| `.gitignore` | 데이터·모델·로그·이미지 등 대용량/생성물 제외 |

> 데이터셋·모델·이미지·로그는 git에 올리지 않습니다. 데이터는 각 머신에서 생성하고, 학습 데이터셋은 Hugging Face로 공유합니다.

## 환경 설정

**수집 (Isaac Sim, Windows standalone)**
- Isaac Sim 5.1
- **numpy 1.26.x 필수** — numpy 2.x는 OmniGraph `unknown dtype size=0` 에러 발생
- `pick_place_collect.py` 상단에서 경로 수정:
  - `ISAACSIM_PATH`, `SAVE_PATH`, `FRANKA_USD`

**변환·학습 (WSL2 또는 Linux 워크스테이션, conda `lerobot` 환경)**
- `pip`/conda로 `lerobot` 설치
- Hugging Face 로그인(`huggingface-cli login`)
- `convert_to_lerobot.py` 상단에서 `SRC`(수집 폴더), `REPO_ID`(HF 데이터셋 이름) 수정

## 목표하는 결과

**학습 때 보지 못한 새로운 큐브 위치에서도 일반화되는** pick-and-place 정책.

- 초기 정책은 "본 적 있는" 위치엔 5mm 이내로 정확히 도달하지만, "새로운" 위치는 최대 20cm까지 빗나가는 일반화 실패를 보였습니다 (사실상 신규 위치 0% 성공).
- 원인은 **큐브 위치를 넓은 연속 공간에 얇게(연속 랜덤) 흩뿌려 수집**한 것이었습니다.
- 해결책으로 `pick_place_collect.py`를 **격자 샘플링(v8)** 으로 재작성:
  - `CUBE_GRID_NX × CUBE_GRID_NY = 5×5 = 25개 지점`
  - 지점당 `EPISODES_PER_POINT = 12`회 반복 (±1cm 지터)
  - → 지점당 데이터 밀도를 확보해 일반화 유도. 타겟(놓을) 위치는 연속 랜덤 유지.

## 현재 단계

- ✅ **v5 데이터 재수집 완료** — 270 에피소드 / 288,090 프레임 (wrist + overhead 카메라, 160×120)
- ✅ **LeRobot 포맷 변환 완료** (로컬)
- ⏳ **HF 업로드 보류** — 사내 프록시(Somansa)의 TLS 가로채기로 WSL에서 `push_to_hub`가 SSL 인증서 오류로 실패. 파이프라인을 **Linux 워크스테이션으로 이전 중**.
- ⏭ **ACT 학습 예정** — 업로드된 HF 데이터셋에서 `lerobot-train` CLI로 진행 (이 repo에는 ACT 학습 스크립트를 두지 않음)

## 주의 (알려진 이슈)

- **HF 업로드 SSL 오류**: 사내망에서는 `CERTIFICATE_VERIFY_FAILED`(Somansa Root CA)로 huggingface.co 업로드가 막힘. Somansa 루트 CA를 신뢰시키거나, 프록시 밖 네트워크에서 업로드해야 함.
- **`bc_train_vision_resnet18.py`는 그대로 재사용 금지**: `REPO_ID`가 옛 데이터셋(`jamongsteak/pickplace_vision`), `PROPRIO_DIM=18`(구 스펙)로 하드코딩되어 있음. 현재 변환 스크립트는 `PROPRIO_DIM=9`(joint_pos만)이므로, BC 경로를 다시 쓰려면 두 값을 먼저 맞춰야 함. 지금은 ACT에 집중하느라 보류.
