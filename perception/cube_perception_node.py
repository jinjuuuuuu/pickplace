#!/usr/bin/env python3
# cube_perception_node.py  (WSL2 / ROS2 Humble)
# ---------------------------------------------------------------------------
# 카메라(RGB + Depth) 기반으로 큐브의 3D 위치를 추정하는 ROS2 노드.
#
# [전체 그림]
#   Isaac Sim(Windows) ──ROS2(DDS)──>  /rgb, /depth, /camera_info
#                                            │
#                                            ▼   (이 노드)
#                              YOLO11 + 색상으로 큐브 픽셀 검출
#                              depth 로 역투영(2D→3D, optical frame)
#                              외부행렬로 카메라→월드 좌표 변환
#                                            │
#                                            ▼
#                       /cube_pose (PointStamped, world frame)  ← RViz/디버그용
#                       (옵션) UDP 로 bc_deploy 에 (x,y,z) 직접 전송
#
# [핵심 아이디어]
#   - depth 영상의 한 픽셀 값 d 는 "그 픽셀이 보는 점까지의 광축(z)방향 거리"(미터).
#   - 핀홀 모델 역투영:  X = (u-cx)/fx * d,  Y = (v-cy)/fy * d,  Z = d   (카메라 optical 좌표계)
#   - 여기에 카메라→월드 4x4 외부행렬(T_world_optical)을 곱하면 월드 좌표가 나온다.
#   - fx, fy, cx, cy 는 /camera_info 에서 자동으로 받는다(하드코딩 불필요).
#
# 실행(WSL2, ROS2 humble 소스 + ultralytics 설치된 환경):
#   python3 cube_perception_node.py
#   # 토픽 이름이 다르면:
#   python3 cube_perception_node.py --ros-args -p rgb_topic:=/rgb -p depth_topic:=/depth
# ---------------------------------------------------------------------------

import socket
import struct
import numpy as np

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy

from sensor_msgs.msg import Image, CameraInfo
from geometry_msgs.msg import PointStamped
from visualization_msgs.msg import Marker

from cv_bridge import CvBridge
import cv2

# YOLO 는 선택 사항(없어도 색상 검출로 동작)
try:
    from ultralytics import YOLO
    _HAS_YOLO = True
except Exception:
    _HAS_YOLO = False


class CubePerceptionNode(Node):
    def __init__(self):
        super().__init__("cube_perception_node")

        # ===== 파라미터 =====================================================
        self.declare_parameter("rgb_topic",        "/rgb")
        self.declare_parameter("depth_topic",      "/depth")
        self.declare_parameter("camera_info_topic","/camera_info")
        self.declare_parameter("cube_pose_topic",  "/cube_pose")

        # ----- 검출기(교체 가능한 백엔드) -----------------------------------
        # detector:
        #   "color" : 빨간 큐브 HSV 색상 검출(학습 불필요, 가장 가벼움)
        #   "yolo"  : 일반/커스텀 YOLO 가중치(큐브 전용 best.pt 학습 시 최적)
        #   "world" : YOLO-World 오픈-보캐뷸러리(학습 불필요, 텍스트 프롬프트로 탐지)
        # 어떤 백엔드든 출력은 동일: (픽셀 u,v + bbox) → 이후 depth→3D 변환은 공유.
        self.declare_parameter("detector", "color")
        # 신경망 검출기가 아무것도 못 찾으면 색상 검출로 보조(yolo/world 모드에서만).
        self.declare_parameter("color_fallback", True)
        self.declare_parameter("yolo_conf",  0.25)
        self.declare_parameter("imgsz",      640)
        # --- "yolo" 백엔드용 ---
        self.declare_parameter("yolo_model", "yolo11n.pt")  # 큐브 전용 best.pt 경로로 교체 가능
        self.declare_parameter("yolo_cube_classes", ["cube", "box", "block"])  # 비우면 최고 conf 박스
        # --- "world" 백엔드용 (오픈-보캐뷸러리) ---
        self.declare_parameter("world_model", "yolov8s-world.pt")  # 자동 다운로드
        self.declare_parameter("prompts", ["red cube", "cube", "box", "block"])

        # ----- 카메라 → 월드 외부행렬 (T_world_optical, 4x4) ------------------
        # bc_deploy_cam.py 에서 카메라를 (0.45, 0.0, 1.2)에 '회전 없이'(top-down) 배치한 경우의 값.
        # optical 좌표계(z=정면/아래, x=영상오른쪽, y=영상아래) → 월드:
        #   world_x = 0.45 + (u-cx)/fx * d
        #   world_y = 0.0  - (v-cy)/fy * d
        #   world_z = 1.2  - d
        # 카메라 위치/자세를 바꾸면 아래 행렬만 그에 맞게 수정하면 된다.
        self.declare_parameter("cam_pos",  [0.45, 0.0, 1.2])   # 카메라 월드 위치
        # 회전행렬 R_world_optical 을 9개 값(row-major)으로. top-down 기본값:
        #   [[1,0,0],[0,-1,0],[0,0,-1]]
        self.declare_parameter("R_world_optical",
                               [1.0, 0.0, 0.0,
                                0.0,-1.0, 0.0,
                                0.0, 0.0,-1.0])

        # ----- UDP 직접 전송(옵션): bc_deploy 가 rclpy 없이 받을 때 사용 ------
        self.declare_parameter("udp_enable", False)
        self.declare_parameter("udp_host",   "127.0.0.1")  # Isaac(Windows) 가 받는 IP
        self.declare_parameter("udp_port",   5599)

        g = lambda n: self.get_parameter(n).value
        self.rgb_topic   = g("rgb_topic")
        self.depth_topic = g("depth_topic")
        self.info_topic  = g("camera_info_topic")
        self.detector       = g("detector")
        self.color_fallback = bool(g("color_fallback"))
        self.yolo_conf      = float(g("yolo_conf"))
        self.imgsz          = int(g("imgsz"))
        self.yolo_cube_classes = [str(c).lower() for c in g("yolo_cube_classes")]

        self.cam_pos = np.array(g("cam_pos"), dtype=np.float64)
        self.R_wo    = np.array(g("R_world_optical"), dtype=np.float64).reshape(3, 3)

        self.udp_enable = bool(g("udp_enable"))
        self.udp_addr   = (g("udp_host"), int(g("udp_port")))
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM) if self.udp_enable else None

        # ===== 상태 =========================================================
        self.bridge = CvBridge()
        self.K = None          # 3x3 intrinsics (camera_info 에서 채움)
        self.fx = self.fy = self.cx = self.cy = None
        self.latest_depth = None       # 최신 depth(미터, float32)
        self._got_rgb = False
        self._got_depth = False

        # ===== 신경망 검출기 로드 (yolo / world) ============================
        self.model = None
        self._load_neural_detector(g)

        # ===== 발행 =========================================================
        self.pub_pose   = self.create_publisher(PointStamped, g("cube_pose_topic"), 10)
        self.pub_marker = self.create_publisher(Marker, "/cube_marker", 10)

        # ===== 구독 =========================================================
        # 이미지 스트림은 BEST_EFFORT 로 받는 게 안전(발행자가 reliable/best_effort 어느 쪽이든 수신 가능)
        qos = QoSProfile(reliability=ReliabilityPolicy.BEST_EFFORT,
                         history=HistoryPolicy.KEEP_LAST, depth=5)
        self.create_subscription(CameraInfo, self.info_topic, self.info_cb, 10)
        # rgb 와 depth 를 따로 받아 각자 최신값을 캐싱(타임스탬프 동기화에 의존 X → 더 안정적)
        self.create_subscription(Image, self.depth_topic, self.depth_cb, qos)
        self.create_subscription(Image, self.rgb_topic,   self.rgb_cb,   qos)
        # 2초마다 데이터 수신 상태 점검(영상이 안 들어오면 원인 안내)
        self.create_timer(2.0, self.health_cb)

        self.get_logger().info(
            f"준비 완료. 구독: {self.rgb_topic}, {self.depth_topic}, {self.info_topic} "
            f"| 검출기={self.detector} | UDP={'on '+str(self.udp_addr) if self.udp_enable else 'off'}")

    # --------------------------------------------------------------------- #
    def info_cb(self, msg: CameraInfo):
        if self.K is None:
            self.K = np.array(msg.k, dtype=np.float64).reshape(3, 3)
            self.fx, self.fy = self.K[0, 0], self.K[1, 1]
            self.cx, self.cy = self.K[0, 2], self.K[1, 2]
            self.get_logger().info(
                f"[camera_info] fx={self.fx:.1f} fy={self.fy:.1f} "
                f"cx={self.cx:.1f} cy={self.cy:.1f} ({msg.width}x{msg.height})")

    # --------------------------------------------------------------------- #
    def depth_cb(self, msg: Image):
        depth = self.bridge.imgmsg_to_cv2(msg, desired_encoding="passthrough")
        depth = np.asarray(depth, dtype=np.float32)
        if msg.encoding in ("16UC1", "mono16"):   # uint16(mm) → 미터
            depth = depth / 1000.0
        self.latest_depth = depth
        if not self._got_depth:
            self._got_depth = True
            self.get_logger().info(
                f"[수신] depth 첫 프레임 OK (enc={msg.encoding}, shape={depth.shape})")

    def rgb_cb(self, msg: Image):
        if not self._got_rgb:
            self._got_rgb = True
            self.get_logger().info(f"[수신] rgb 첫 프레임 OK (enc={msg.encoding})")
        # rgb 가 들어올 때마다 가장 최근 depth 로 처리
        if self.K is None or self.latest_depth is None:
            return
        rgb = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        self.process(rgb, self.latest_depth, msg.header.stamp)

    def process(self, rgb, depth, stamp):
        det = self.detect_cube(rgb)          # (u, v, bbox) 또는 None
        if det is None:
            self.get_logger().warn("큐브 미검출(영상은 들어옴) — 색상/조명/시야 확인",
                                   throttle_duration_sec=2.0)
            return
        u, v, bbox = det

        # depth 해상도가 rgb 와 다를 경우 픽셀 좌표 스케일 보정
        if depth.shape[1] != rgb.shape[1] or depth.shape[0] != rgb.shape[0]:
            sx = depth.shape[1] / rgb.shape[1]
            sy = depth.shape[0] / rgb.shape[0]
            u_d, v_d = u * sx, v * sy
            bbox_d = (int(bbox[0]*sx), int(bbox[1]*sy), int(bbox[2]*sx), int(bbox[3]*sy))
        else:
            u_d, v_d, bbox_d = u, v, bbox

        d = self.sample_depth(depth, u_d, v_d, bbox_d)
        if d is None or not np.isfinite(d) or d <= 0.0:
            self.get_logger().warn("큐브는 찾았으나 그 위치 depth 값이 무효",
                                   throttle_duration_sec=2.0)
            return

        # 2D → 3D (optical 좌표계). K 는 rgb 기준이므로 rgb 픽셀 좌표(u,v) 사용
        x_o = (u - self.cx) / self.fx * d
        y_o = (v - self.cy) / self.fy * d
        p_opt = np.array([x_o, y_o, d], dtype=np.float64)
        p_world = self.R_wo @ p_opt + self.cam_pos
        self.publish(p_world, stamp)

    def health_cb(self):
        if not (self._got_rgb and self._got_depth and self.K is not None):
            self.get_logger().warn(
                f"영상 대기중... rgb={'O' if self._got_rgb else 'X'} "
                f"depth={'O' if self._got_depth else 'X'} "
                f"info={'O' if self.K is not None else 'X'}  "
                f"→ Isaac에서 bc_deploy_cam.py 실행+시뮬 Play 확인, "
                f"WSL2에서 'ros2 topic hz {self.rgb_topic}' 로 발행여부 점검")

    # --------------------------------------------------------------------- #
    def _load_neural_detector(self, g):
        """detector 가 yolo/world 면 모델을 로드. 실패 시 color 로 강등."""
        if self.detector == "color":
            self.get_logger().info("[검출기] color (HSV 빨강)")
            return
        if not _HAS_YOLO:
            self.get_logger().warn("[검출기] ultralytics 미설치 → color 로 전환")
            self.detector = "color"
            return
        try:
            if self.detector == "world":
                from ultralytics import YOLOWorld
                self.model = YOLOWorld(g("world_model"))
                self.prompts = [str(p) for p in g("prompts")]
                self.model.set_classes(self.prompts)        # 오픈-보캐뷸러리: 텍스트로 클래스 지정
                self.get_logger().info(
                    f"[검출기] world: {g('world_model')} 프롬프트={self.prompts}")
            elif self.detector == "yolo":
                from ultralytics import YOLO
                self.model = YOLO(g("yolo_model"))
                names = [str(v).lower() for v in self.model.names.values()]
                self.get_logger().info(
                    f"[검출기] yolo: {g('yolo_model')} (클래스 {len(names)}개)")
                if self.yolo_cube_classes and not any(c in names for c in self.yolo_cube_classes):
                    self.get_logger().warn(
                        "[검출기] 이 YOLO 모델엔 cube 계열 클래스가 없습니다(예: COCO 사전학습). "
                        "큐브 전용 best.pt 를 학습해 yolo_model 로 지정하거나 detector:=world/color 사용.")
            else:
                self.get_logger().warn(f"[검출기] 알 수 없는 detector='{self.detector}' → color")
                self.detector = "color"
        except Exception as e:
            self.get_logger().warn(f"[검출기] 로드 실패 → color 로 전환: {e}")
            self.detector = "color"
            self.model = None

    # --------------------------------------------------------------------- #
    def detect_cube(self, bgr):
        """큐브 픽셀 중심(u,v)과 bbox(x1,y1,x2,y2) 반환. 실패 시 None.
        백엔드(color/yolo/world)와 무관하게 동일 포맷을 반환한다."""
        if self.model is not None:                  # yolo / world
            det = self._detect_neural(bgr)
            if det is not None:
                return det
            if not self.color_fallback:
                return None
        return self._detect_color(bgr)              # color 또는 신경망 폴백

    def _detect_neural(self, bgr):
        res = self.model.predict(bgr, conf=self.yolo_conf, imgsz=self.imgsz, verbose=False)[0]
        best, best_score = None, 0.0
        for b in res.boxes:
            name = str(self.model.names[int(b.cls)]).lower()
            # 커스텀 yolo 모델은 cube 계열만 채택(world 는 set_classes 로 이미 제한됨)
            if self.detector == "yolo" and self.yolo_cube_classes and name not in self.yolo_cube_classes:
                continue
            score = float(b.conf)
            if score > best_score:
                x1, y1, x2, y2 = b.xyxy[0].tolist()
                best_score = score
                best = (x1, y1, x2, y2)
        if best is None:
            return None
        x1, y1, x2, y2 = best
        return ((x1 + x2) / 2.0, (y1 + y2) / 2.0, (int(x1), int(y1), int(x2), int(y2)))

    def _detect_color(self, bgr):
        # 빨간 큐브 — HSV 두 구간(빨강은 hue 가 0/180 양끝으로 갈라짐)
        hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
        m1 = cv2.inRange(hsv, (0, 90, 60),   (10, 255, 255))
        m2 = cv2.inRange(hsv, (170, 90, 60), (180, 255, 255))
        mask = cv2.morphologyEx(m1 | m2, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
        cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not cnts:
            return None
        c = max(cnts, key=cv2.contourArea)
        if cv2.contourArea(c) < 60:   # 너무 작은 잡음 무시
            return None
        x, y, w, h = cv2.boundingRect(c)
        M = cv2.moments(c)
        u = M["m10"] / (M["m00"] + 1e-6)
        v = M["m01"] / (M["m00"] + 1e-6)
        return (u, v, (x, y, x + w, y + h))

    # --------------------------------------------------------------------- #
    def sample_depth(self, depth, u, v, bbox):
        """bbox 중앙 주변의 유효 depth 중앙값(미터). 단일 픽셀 노이즈에 강함."""
        x1, y1, x2, y2 = bbox
        # bbox 중앙 1/3 영역만 사용(가장자리/배경 깊이 섞임 방지)
        cx1 = int(x1 + (x2 - x1) * 0.33); cx2 = int(x1 + (x2 - x1) * 0.67)
        cy1 = int(y1 + (y2 - y1) * 0.33); cy2 = int(y1 + (y2 - y1) * 0.67)
        cx1, cx2 = max(cx1, 0), min(cx2, depth.shape[1])
        cy1, cy2 = max(cy1, 0), min(cy2, depth.shape[0])
        patch = depth[cy1:cy2, cx1:cx2]
        vals = patch[np.isfinite(patch) & (patch > 0.0)]
        if vals.size == 0:
            # 폴백: 중심 픽셀 하나
            iv, iu = int(round(v)), int(round(u))
            if 0 <= iv < depth.shape[0] and 0 <= iu < depth.shape[1]:
                d = depth[iv, iu]
                return float(d) if np.isfinite(d) and d > 0 else None
            return None
        return float(np.median(vals))

    # --------------------------------------------------------------------- #
    def publish(self, p_world, stamp):
        msg = PointStamped()
        msg.header.stamp = stamp
        msg.header.frame_id = "world"
        msg.point.x, msg.point.y, msg.point.z = map(float, p_world)
        self.pub_pose.publish(msg)

        mk = Marker()
        mk.header.frame_id = "world"
        mk.header.stamp = stamp
        mk.ns = "cube"; mk.id = 0
        mk.type = Marker.CUBE; mk.action = Marker.ADD
        mk.pose.position.x, mk.pose.position.y, mk.pose.position.z = map(float, p_world)
        mk.pose.orientation.w = 1.0
        mk.scale.x = mk.scale.y = mk.scale.z = 0.05
        mk.color.r, mk.color.g, mk.color.b, mk.color.a = 0.9, 0.1, 0.1, 0.9
        self.pub_marker.publish(mk)

        if self.sock is not None:
            # 단순 포맷: 3개의 float64 (little-endian)
            self.sock.sendto(struct.pack("<3d", *map(float, p_world)), self.udp_addr)

        self.get_logger().info(
            f"cube(world) = [{p_world[0]:+.3f}, {p_world[1]:+.3f}, {p_world[2]:+.3f}] m",
            throttle_duration_sec=1.0)


def main():
    rclpy.init()
    node = CubePerceptionNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
