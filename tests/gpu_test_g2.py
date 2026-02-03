"""G2: 기존 test_mimic_video.py가 깨지지 않는지 확인"""
import subprocess, sys
result = subprocess.run(
    ["conda", "run", "-n", "mimic_video", "pytest", "tests/test_mimic_video.py", "-v"],
    cwd="/rlwrld3/home/hojin/vam_workspace/mimic-video",
    capture_output=False,
)
sys.exit(result.returncode)
