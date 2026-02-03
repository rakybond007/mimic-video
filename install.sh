#!/bin/bash
# mimic-video 설치 스크립트 (conda + pip)
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"
ENV_NAME="mimic_video"

echo "=== 1. Conda 환경 생성 ==="
if conda env list | grep -q "^${ENV_NAME} "; then
  echo "환경 ${ENV_NAME} 이(가) 이미 존재합니다."
else
  conda env create -f environment.yml
fi

echo ""
echo "=== 2. pip 의존성 및 mimic-video 설치 ==="
conda run -n "$ENV_NAME" pip install --upgrade pip
conda run -n "$ENV_NAME" pip install -r requirements.txt
conda run -n "$ENV_NAME" pip install -e .

echo ""
echo "설치 완료. 사용: conda activate $ENV_NAME"
echo "테스트 포함 설치: pip install -e '.[test]'"
