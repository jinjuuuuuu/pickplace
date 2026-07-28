# 카메라 기반 큐브 위치 추정 (YOLO + Depth) — 연결 가이드

BC 배포 시 큐브 좌표를 **시뮬레이터 정답(`cube.get_world_pose()`) 대신 카메라로 추정**하도록 바꾼 구성입니다.

```
Isaac Sim (Windows)                          WSL2 (ROS2 Humble)
┌───────────────────────────┐   /rgb,/depth   ┌────────────────────────────┐
│ bc_deploy_cam.py          │  /camera_info   │ cube_perception_node.py    │
│  · top-down 카메라 추가     │ ──────────────▶ │  · YOLO11 + 색상으로 큐브 검출 │
│  · ROS2로 영상 발행         │                 │  · depth 역투영 → 3D        │
│  · 큐브좌표 수신 → obs 주입  │ ◀────────────── │  · 외부행렬로 world 변환     │
└───────────────────────────┘  /cube_pose 또는 │  · /cube_pose 발행          │
                                  UDP(x,y,z)    └────────────────────────────┘
```

핵심: BC 모델의 입력(obs) 자체는 그대로입니다. 바뀐 건 obs 안의 `cube_world_pos`를 채우는 **출처**뿐입니다.

---

## 1. 사전 준비 (WSL2)

```bash
# ROS2 humble 환경에서
pip install ultralytics opencv-python        # YOLO + 영상처리
sudo apt install ros-humble-cv-bridge ros-humble-vision-msgs   # cv_bridge, 메시지
```

`cube_perception_node.py`를 WSL2 쪽으로 복사해두세요(예: `~/cube_perception_node.py`).
Isaac(Windows) ↔ WSL2 간 ROS2 통신은 기존에 쓰던 `fastdds.xml` 설정을 그대로 사용합니다.

---

## 2. 실행 순서

**① Isaac (Windows, PowerShell)**
```powershell
"C:\isaacsim\python.bat" "C:\Users\user\Desktop\claude_jetbot\bc_deploy_cam.py"
```
→ 카메라를 배치하고 `/rgb /depth /camera_info`를 발행하기 시작합니다.
   (`USE_CAMERA_CUBE=True`라 첫 추정값이 올 때까지 최대 ~10초 대기합니다.)

**② Perception 노드 (WSL2)** — 큐브좌표를 **UDP로 직접** 보내는 기본 방식:
```bash
python3 cube_perception_node.py --ros-args \
  -p udp_enable:=true \
  -p udp_host:=<WINDOWS_IP> \
  -p udp_port:=5599
```
`<WINDOWS_IP>` = WSL2에서 본 Windows 호스트 IP. 보통:
```bash
ip route show | grep -i default | awk '{print $3}'   # 기본 게이트웨이(대개 Windows 호스트)
```
Windows 11 미러드 네트워킹이면 `udp_host:=127.0.0.1`로도 됩니다.

토픽 이름이 다르면 `-p rgb_topic:=/rgb -p depth_topic:=/depth -p camera_info_topic:=/camera_info`로 맞춰주세요.

---

## 3. 큐브좌표 수신 방식 두 가지 (`bc_deploy_cam.py`의 `RECEIVE_MODE`)

| 모드 | 설명 | 장점 / 주의 |
|---|---|---|
| `"udp"` (기본) | perception 노드가 (x,y,z)를 UDP로 직접 전송 | **추가 설치 불필요·가장 안정적.** 노드를 `udp_enable:=true`로 실행해야 함 |
| `"ros2"` | `bc_deploy_cam.py`가 `/cube_pose`를 직접 구독 | Isaac 파이썬에서 `rclpy` import가 되는 경우만. 안 되면 자동으로 안내 메시지 출력 → `"udp"`로 전환 |

`"ros2"`로 쓰려면 perception 노드는 그냥 `udp_enable` 없이 실행하면 됩니다(`/cube_pose`는 항상 발행됨).

---

## 4. 카메라 내부/외부 파라미터

**내부(intrinsics, K)** — 자동입니다. Isaac의 CameraInfoHelper가
`fx = W·focal/h_aper`, `fy = H·focal/v_aper`, `cx=W/2`, `cy=H/2`로 계산해
`/camera_info`로 발행하고, 노드가 그대로 읽습니다. 손으로 넣을 필요 없습니다.

**외부(extrinsics, 카메라→월드)** — `bc_deploy_cam.py`에서 카메라를
`(0.45, 0, 1.2)`에 **회전 없이**(수직 하방) 두었기 때문에, 노드의 기본 외부행렬과 정확히 일치합니다:

```
world_x = 0.45 + (u-cx)/fx · d
world_y = 0.0  − (v-cy)/fy · d
world_z = 1.2  − d            (d = depth = 광축방향 거리, 미터)
```

→ 왕복 투영 테스트에서 오차 ≈ 0 (완전 일치) 확인했습니다.

**카메라 위치/자세를 바꾸려면** `bc_deploy_cam.py`의 `CAM_POS`와, 노드의 두 파라미터
`cam_pos`, `R_world_optical`(3×3, row-major)만 같은 값으로 맞추면 됩니다. top-down 기본값:
```
cam_pos          = [0.45, 0.0, 1.2]
R_world_optical  = [1,0,0, 0,-1,0, 0,0,-1]
```

---

## 5. 검출기(detector) — 교체 가능한 3개 백엔드

검출기는 **무엇을 쓰든 출력이 동일**합니다: 큐브의 픽셀 (u,v)+bbox. 그 뒤 depth→3D→world 변환은
백엔드와 무관하게 공유됩니다. 그래서 자유롭게 갈아끼울 수 있습니다.

| `detector` | 방식 | 학습 | 비고 |
|---|---|---|---|
| `color` (기본) | 빨간색 HSV 색상 검출 | 불필요 | 가장 가볍고, 깔끔한 시뮬 환경에서 안정적 |
| `yolo` | 일반/커스텀 YOLO 가중치 | 커스텀 시 필요 | 큐브 전용 `best.pt` 학습 시 최적. **사전학습 `yolo11n.pt`는 COCO라 'cube' 클래스 없음** |
| `world` | YOLO-World 오픈-보캐뷸러리 | 불필요 | **텍스트 프롬프트**로 탐지(`"red cube"` 등). 추상 도형은 명중률 들쭉날쭉할 수 있음 |

> COCO 80개 클래스에는 'cube'가 없어서, 사전학습 YOLO만으로는 큐브를 못 잡습니다.
> 그래서 `yolo`/`world` 모드에서 신경망이 아무것도 못 찾으면 `color_fallback`(기본 True)으로 색상 검출이 받쳐줍니다.

실행 예시:
```bash
# 색상 (기본, 가장 가벼움)
python3 cube_perception_node.py --ros-args -p detector:=color

# YOLO-World — 학습 없이 텍스트 프롬프트로 시험
python3 cube_perception_node.py --ros-args \
  -p detector:=world -p prompts:='["red cube","cube","box"]'

# 커스텀 학습 모델 (나중에 best.pt 생기면)
python3 cube_perception_node.py --ros-args \
  -p detector:=yolo -p yolo_model:=/path/best.pt -p color_fallback:=false
```

미래 확장: 실세계 JetBot처럼 배경이 복잡해지면 `world`로 빠르게 실험하거나 `best.pt`를 학습해
`yolo`로 교체하면 됩니다. 코드/배포 파이프라인은 그대로 두고 `detector` 한 줄만 바꾸면 됩니다.
(참고로 CUDA 드라이버가 구형이면 YOLO 계열은 CPU로 돌아 느립니다 — 동작엔 문제없음.)

---

## 6. 검증 / 캘리브레이션 체크

- 노드 로그에 `cube(world) = [x, y, z]`가 1초마다 찍힙니다.
  이 값이 실제 스폰 위치(`CUBE_POS ≈ [0.40, 0.15, 0.025]`)와 가까우면 정상입니다.
- RViz2에서 `/cube_marker`(빨간 큐브)와 `/rgb` 이미지로 눈으로 확인할 수 있습니다.
- **만약 카메라가 수직 하방을 안 보고 옆을 본다면** (Isaac 버전에 따라 기본 카메라 자세가 다를 수 있음),
  `bc_deploy_cam.py`에서 `cam_xform.SetRotate(Gf.Vec3f(0,0,0))`을 `(-90,0,0)` 등으로 바꿔
  뷰포트에서 작업대가 정면에 보이게 맞춘 뒤, 그에 맞게 `R_world_optical`을 수정하세요.
- depth 영상이 �black/white로만 보이면 FOV에 무한원(바닥 너머)이 들어간 것 → 클리핑/시야 조정.

---

## 7. 디버그 팁

```bash
ros2 topic list                       # /rgb /depth /camera_info /cube_pose 보이는지
ros2 topic echo /camera_info --once   # K(fx,fy,cx,cy) 확인
ros2 run rqt_image_view rqt_image_view  # /rgb, /depth 영상 확인
ros2 topic echo /cube_pose            # 추정 좌표 확인
```
