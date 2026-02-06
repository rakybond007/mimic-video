#!/usr/bin/env python3
"""Quick data validation script."""
import torch
from mimic_video.data.data_config import LiberoDataConfig
from mimic_video.data.dataset import LeRobotLiberoDataset

data_config = LiberoDataConfig(num_frames=8, video_resolution=224, num_future_frames=1, future_frame_start=16)
print(f"observation_indices: {data_config.observation_indices}")
print(f"future_video_indices: {data_config.future_video_indices}")

ds = LeRobotLiberoDataset(
    dataset_path="/sjw_alinlab2/home/myungkyu/workspace/AlinVLA/.cache/huggingface/lerobot/kimtaey/libero_gr00t_delta",
    data_config=data_config, max_state_dim=64, max_action_dim=32, training=True
)

sample = ds[1000]
v = sample["video"]
fv = sample["future_video"]
js = sample["joint_state"]
a = sample["actions"]
am = sample["action_mask"]
p = sample["prompt"]

print(f"\nvideo: shape={v.shape}, min={v.min():.4f}, max={v.max():.4f}, mean={v.mean():.4f}")
print(f"future_video: shape={fv.shape}, min={fv.min():.4f}, max={fv.max():.4f}, mean={fv.mean():.4f}")
print(f"joint_state: shape={js.shape}, min={js.min():.4f}, max={js.max():.4f}, mean={js.mean():.4f}")
print(f"actions: shape={a.shape}, min={a.min():.4f}, max={a.max():.4f}, mean={a.mean():.4f}")
print(f"action_mask: shape={am.shape}, sum={am.sum().item()}")
print(f"prompt: {p[:80]}...")

# Check non-zero
print("\n=== Non-zero Check ===")
print(f"video all zeros: {(v == 0).all().item()}")
print(f"future_video all zeros: {(fv == 0).all().item()}")
print(f"joint_state all zeros: {(js == 0).all().item()}")
print(f"actions all zeros: {(a == 0).all().item()}")
