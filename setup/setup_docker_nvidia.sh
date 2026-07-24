#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# setup_docker_nvidia.sh
# NVIDIA Container Toolkit 설치 + Docker GPU 연동 구성 + 검증
#
# 전제: Docker Engine은 이미 설치돼 있음. Ubuntu, sudo(관리자) 권한 필요.
# 사용법:
#   cd ~/pickplace && git pull
#   bash setup/setup_docker_nvidia.sh
# ---------------------------------------------------------------------------
set -eu

echo "==> [0/5] 사전 확인"
if ! command -v docker >/dev/null 2>&1; then
  echo "!! docker가 없습니다. 먼저 Docker Engine을 설치하세요:"
  echo "     curl -fsSL https://get.docker.com | sudo sh"
  exit 1
fi
docker --version
if ! command -v nvidia-smi >/dev/null 2>&1; then
  echo "!! nvidia-smi가 없습니다. NVIDIA 드라이버를 먼저 확인하세요."
  exit 1
fi
nvidia-smi | head -3

echo "==> [1/5] NVIDIA Container Toolkit APT 저장소 등록"
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey \
  | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list \
  | sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' \
  | sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list >/dev/null

echo "==> [2/5] 설치"
sudo apt-get update
sudo apt-get install -y nvidia-container-toolkit

echo "==> [3/5] Docker 런타임에 NVIDIA 연동 구성 + 재시작"
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker

echo "==> [4/5] 현재 사용자($USER)를 docker 그룹에 추가 (sudo 없이 docker 쓰려면)"
sudo usermod -aG docker "$USER"
echo "   ※ 그룹 반영은 '재로그인' 또는 'newgrp docker' 후 적용됩니다."

echo "==> [5/5] GPU 연동 검증 (CUDA 베이스 이미지에서 nvidia-smi 실행)"
echo "   이미지를 처음 받으면 다운로드에 시간이 걸립니다..."
sudo docker run --rm --gpus all nvidia/cuda:12.8.0-base-ubuntu22.04 nvidia-smi | head -15

echo
echo "✅ 완료! 위에 호스트와 동일한 GPU 표가 컨테이너 안에서 떴다면 성공입니다."
echo "   다음 단계: NGC 로그인"
echo "     docker login nvcr.io   (Username: \$oauthtoken  /  Password: NGC API key)"
