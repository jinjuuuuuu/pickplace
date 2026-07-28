try:
    from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
except ImportError:
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

print("로컬 데이터셋을 불러오는 중...")
# 수정된 새로운 로컬 폴더명을 넣어줍니다.
dataset = LeRobotDataset("jamongsteak/pickplace_vision_v2")

print("허깅페이스 허브로 업로드 시작!")
dataset.push_to_hub()

print("업로드 완료!")