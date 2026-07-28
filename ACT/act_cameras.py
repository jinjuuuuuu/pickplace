# act_cameras.py  —  Isaac Sim 카메라 생성/캡처 (수집·배포 공용)
# ---------------------------------------------------------------------------
# 크래시 회피 핵심:
#   1) Camera 클래스가 '직접 프림을 생성'하게 한다(올바른 render 속성/intrinsic 설정).
#      - 미리 만든 빈 USD Camera 프림을 wrap하면 size=0 dtype 크래시가 난다.
#   2) 모두 '고정' 카메라(articulation 자식 X) -> render product 안정.
#   3) 초기화 순서: 카메라 생성 -> world.reset() -> app.update 워밍업
#                  -> camera.initialize() -> world.step 워밍업 -> get_rgba()
# ---------------------------------------------------------------------------
import numpy as np
from isaacsim.sensors.camera import Camera
import isaacsim.core.utils.numpy.rotations as rot_utils


def create_cameras(cfg):
    """Camera 객체들을 생성(프림도 함께 생성). world.reset() '전에' 호출 권장."""
    cams = {}
    specs = {
        "top":   (cfg.OVERHEAD_CAM_POS, cfg.OVERHEAD_CAM_EULER, "/World/ACTTopCam"),
        "front": (cfg.FRONT_CAM_POS,    cfg.FRONT_CAM_EULER,    "/World/ACTFrontCam"),
    }
    for name in cfg.CAMERA_NAMES:
        pos, euler, prim = specs[name]
        cams[name] = Camera(
            prim_path=prim,
            position=np.array(pos, dtype=float),
            orientation=rot_utils.euler_angles_to_quats(
                np.array(euler, dtype=float), degrees=True),
            resolution=(cfg.CAM_WIDTH, cfg.CAM_HEIGHT),
        )
    return cams


def init_cameras(cams, world, simulation_app, warmup=30):
    """world.reset() 이후 호출. 렌더 워밍업 -> initialize -> 추가 워밍업."""
    for _ in range(warmup):
        simulation_app.update()
    for cam in cams.values():
        cam.initialize()
    for _ in range(warmup):
        world.step(render=True)


def grab_rgb(cam, cfg):
    """(H,W,3) uint8 RGB 안전 캡처."""
    try:
        rgba = cam.get_rgba()
    except Exception:
        rgba = None
    if rgba is None or np.asarray(rgba).size == 0:
        return np.zeros((cfg.CAM_HEIGHT, cfg.CAM_WIDTH, 3), dtype=np.uint8)
    arr = np.asarray(rgba)[..., :3]
    if arr.dtype != np.uint8:
        arr = (np.clip(arr, 0.0, 1.0) * 255).astype(np.uint8) if arr.max() <= 1.0 \
              else arr.astype(np.uint8)
    if arr.shape[0] != cfg.CAM_HEIGHT or arr.shape[1] != cfg.CAM_WIDTH:
        import cv2
        arr = cv2.resize(arr, (cfg.CAM_WIDTH, cfg.CAM_HEIGHT))
    return np.ascontiguousarray(arr, dtype=np.uint8)
