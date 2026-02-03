"""G4 pre-check: Load one sample from real LIBERO dataset."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

DATASET_PATH = "/sjw_alinlab2/home/myungkyu/workspace/AlinVLA/.cache/huggingface/lerobot/kimtaey/libero_gr00t_delta"

from mimic_video.data.dataset import LeRobotLiberoDataset

print("Loading dataset...")
ds = LeRobotLiberoDataset(DATASET_PATH, num_frames=1, video_resolution=224)
print(f"Dataset length: {len(ds)}")

print("\nLoading sample 0...")
sample = ds[0]
for k, v in sample.items():
    if hasattr(v, 'shape'):
        print(f"  {k}: shape={v.shape}, dtype={v.dtype}")
    else:
        print(f"  {k}: {repr(v)[:80]}")

# Verify expected shapes
video = sample["video"]
actions = sample["actions"]
action_mask = sample["action_mask"]
joint_state = sample["joint_state"]

print(f"\nExpected shapes:")
print(f"  video: (2, 1, 3, 224, 224) → got {tuple(video.shape)}")
print(f"  actions: (16, 32) → got {tuple(actions.shape)}")
print(f"  action_mask: (32,) → got {tuple(action_mask.shape)}")
print(f"  joint_state: (1, 64) → got {tuple(joint_state.shape)}")
print(f"  action_mask True count: {action_mask.sum().item()} (expect 7)")

assert video.shape == (2, 1, 3, 224, 224), f"video shape mismatch: {video.shape}"
assert actions.shape == (16, 32), f"actions shape mismatch: {actions.shape}"
assert action_mask.shape == (32,), f"action_mask shape mismatch: {action_mask.shape}"
assert action_mask.sum().item() == 7, f"action_mask true count: {action_mask.sum().item()}"
assert joint_state.shape[1] == 64, f"joint_state dim mismatch: {joint_state.shape}"
assert sample["prompt"] != "", "prompt should not be empty"

print(f"\n  prompt: {sample['prompt']}")
print("\nData check PASSED")
