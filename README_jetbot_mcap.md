# JetBot → ROS 2 → mcap 자동 기록 실습

Isaac Sim 5.x(Windows) + ROS 2(WSL2) 환경에서, **JetBot을 자동 주행시키며 카메라 데이터를 mcap으로 자동 기록**하는 실습입니다.

```
[Windows] jetbot_record_demo.py  ──토픽 발행──▶  [WSL2] record_jetbot.launch.py ──▶ jetbot_run/*.mcap
 (Isaac Sim 자동 주행)                              (rosbag2 자동 기록·자동 종료)
```

핵심: Isaac Sim은 **토픽을 발행만** 하고, mcap 기록은 **WSL2의 rosbag2**가 합니다. 두 파일을 합쳐 양쪽을 자동화합니다.

---

## 0. 사전 준비

- Isaac Sim 5.x (Windows) 설치 완료
- WSL2 + ROS 2 (Humble 또는 Jazzy) 설치, Isaac Sim ROS 2 워크스페이스 환경 구성
- `isaacsim.ros2.bridge` 확장 사용 가능 상태
- Windows ↔ WSL2 네트워크/DDS 설정 완료 (Isaac Sim 문서의 ROS 2 설치 절차)
- mcap 저장 플러그인 (보통 기본 포함, 없으면 설치):
  ```bash
  sudo apt install ros-$ROS_DISTRO-rosbag2-storage-mcap
  ```

---

## 1. 실행 순서 (중요: record를 먼저 켜고 SIM을 실행)

**① WSL2 터미널** — ROS 2 환경 source 후 기록 launch 실행 (기록 대기 시작)
```bash
source /opt/ros/$ROS_DISTRO/setup.bash      # + 필요 시 워크스페이스 source
ros2 launch ./record_jetbot.launch.py
```

**② Windows 터미널(PowerShell/cmd)** — Isaac Sim 설치 폴더에서 standalone 실행
```bat
python.bat C:\path\to\jetbot_record_demo.py
```

순서가 중요한 이유: rosbag2(구독자)가 먼저 떠 있어야 JetBot이 발행하는 첫 프레임부터 빠짐없이 기록됩니다.

기본 설정이면 약 30초 후 양쪽이 자동 종료되고, WSL2의 작업 폴더에 `jetbot_run/` 디렉터리(= `jetbot_run_0.mcap` + `metadata.yaml`)가 생성됩니다.

---

## 2. 결과 확인 (WSL2)
```bash
ros2 bag info jetbot_run
# Storage id: mcap, /jetbot/camera/rgb, /clock 등이 보이면 성공

# 재생 + 시각화
ros2 bag play jetbot_run
ros2 run rqt_image_view rqt_image_view   # /jetbot/camera/rgb 선택
```

기록 전에 토픽이 실제로 들어오는지 먼저 보고 싶다면:
```bash
ros2 topic list
ros2 topic hz /jetbot/camera/rgb
```

---

## 3. 자주 막히는 부분

- **토픽이 WSL2에서 안 보임**: Windows↔WSL2 DDS/네트워크 설정 문제. `RMW_IMPLEMENTATION`, Fast DDS 프로파일(`FASTRTPS_DEFAULT_PROFILES_FILE`), WSL2 IP 설정을 Isaac Sim 문서대로 다시 확인. 대역폭 큰 이미지 토픽은 특히 영향이 큼.
- **`DOF 이름` 출력과 바퀴가 안 맞음**: 콘솔에 찍힌 `JetBot DOF 이름`을 보고 `jetbot_record_demo.py`의 `WHEEL_DOFS`를 실제 이름으로 수정. (이름이 안 맞으면 'wheel' 포함 조인트를 자동 탐색하도록 되어 있음)
- **카메라 prim을 못 찾음**: JetBot 에셋 구조가 다른 경우. 콘솔의 `카메라 prim` 경로를 확인하고, 필요하면 `find_camera_prim` 대신 정확한 경로를 직접 지정.
- **OmniGraph 노드 타입 에러**: 4.0 이하 버전이면 `isaacsim.*` → `omni.isaac.*` 네임스페이스로 교체 필요.
- **mcap이 아니라 .db3로 저장됨**: `-s mcap` 옵션 누락. launch의 `cmd`에 포함되어 있는지 확인.

---

## 4. 한 줄로 묶기 (선택)

Windows standalone 스크립트가 시작될 때 WSL2 기록까지 같이 띄우고 싶다면, `jetbot_record_demo.py` 상단(SimulationApp 생성 전)에 다음을 추가할 수 있습니다. 단, 환경 차이로 깨지기 쉬워 **권장 기본값은 위의 2-터미널 방식**입니다.

```python
import subprocess
subprocess.Popen([
    "wsl", "-e", "bash", "-lic",
    "source /opt/ros/$ROS_DISTRO/setup.bash && "
    "ros2 bag record -s mcap -o jetbot_run /jetbot/camera/rgb /clock"
])
```
종료 시점 동기화가 까다로우므로, 실습 단계에서는 launch 파일의 자동 종료(`RECORD_SECONDS`)를 쓰는 편이 안전합니다.

---

## 5. 더 기록하고 싶을 때

- **depth/카메라 인트린식**: 그래프에 `ROS2CameraHelper`(type `depth`)와 카메라 인포 노드를 추가하고, launch `TOPICS`에 `/jetbot/camera/depth`, `/jetbot/camera/camera_info` 추가.
- **TF**: `ROS2PublishTransformTree` 노드를 추가하고 `TOPICS`에 `/tf` 추가.
- **압축/처리량 튜닝**: `ros2 bag record -s mcap --storage-config-file mcap_writer_options.yml` 로 청크 크기·압축(zstd) 조절.
