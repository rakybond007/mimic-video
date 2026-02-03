"""Diagnose high loss: check intermediate value scales in the forward pass."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
import torch.nn.functional as F
from mimic_video.cosmos_predict import Cosmos2PredictWrapper
from mimic_video.mimic_video import MimicVideo
from mimic_video.data.dataset import LeRobotLiberoDataset

DATASET_PATH = "/sjw_alinlab2/home/myungkyu/workspace/AlinVLA/.cache/huggingface/lerobot/kimtaey/libero_gr00t_delta"

print("=" * 60)
print("Loss Diagnostic")
print("=" * 60)

# 1. Check raw action/state scales from dataset
print("\n=== 1. Raw Data Scales ===")
dataset = LeRobotLiberoDataset(
    dataset_path=DATASET_PATH, num_frames=1, video_resolution=224,
    max_state_dim=64, max_action_dim=32, training=True,
)
sample = dataset[0]
actions_raw = sample["actions"]  # (16, 32)
mask = sample["action_mask"]     # (32,)
state_raw = sample["joint_state"]  # (1, 64)

real_actions = actions_raw[:, mask]  # (16, 7)
print(f"Actions (7 real dims): min={real_actions.min():.4f}, max={real_actions.max():.4f}, "
      f"mean={real_actions.mean():.4f}, std={real_actions.std():.4f}")
print(f"Actions (padded 32d):  min={actions_raw.min():.4f}, max={actions_raw.max():.4f}, "
      f"std={actions_raw.std():.4f}")
print(f"Joint state (7 real):  min={state_raw[0,:7].min():.4f}, max={state_raw[0,:7].max():.4f}, "
      f"std={state_raw[0,:7].std():.4f}")

# Check a few samples
for i in [0, 100, 1000, 5000]:
    s = dataset[i]
    ra = s["actions"][:, s["action_mask"]]
    print(f"  Sample {i}: action range [{ra.min():.4f}, {ra.max():.4f}], std={ra.std():.4f}")

# 2. Check flow matching scales
print("\n=== 2. Flow Matching Scales ===")
actions_t = actions_raw.unsqueeze(0).cuda()  # (1, 16, 32)

noise = torch.randn_like(actions_t)
flow_target = actions_t - noise
print(f"noise:       std={noise.std():.4f}")
print(f"flow target: std={flow_target.std():.4f}, mean_abs={flow_target.abs().mean():.4f}")

# Check time sampling distribution
from mimic_video.mimic_video import default_sample_time_fn
times = torch.rand(10000)
sampled_times = default_sample_time_fn(times)
print(f"\nTime sampling: mean={sampled_times.mean():.4f}, "
      f"min={sampled_times.min():.4f}, max={sampled_times.max():.4f}")
print(f"  (1-t) denominator: mean={(1-sampled_times).mean():.6f}, "
      f"min={(1-sampled_times).min():.6f}, median={(1-sampled_times).median():.6f}")
pct_close = (sampled_times > 0.99).float().mean()
print(f"  % of time > 0.99: {pct_close*100:.1f}%")
pct_close2 = (sampled_times > 0.95).float().mean()
print(f"  % of time > 0.95: {pct_close2*100:.1f}%")

# 3. Simulate model_output_clean=True scaling
print("\n=== 3. x0-prediction Scaling Impact ===")
t = sampled_times.cuda()
random_pred = torch.randn(1, 16, 32, device="cuda")  # untrained model output
clean_target = actions_t
for t_val in [0.1, 0.5, 0.9, 0.95, 0.99, 0.999]:
    denom = max(1.0 - t_val, 1e-5)
    pred_flow_scale = (random_pred - clean_target).abs().mean() / denom
    flow_target_scale = flow_target.abs().mean()
    mse = ((random_pred - clean_target) / denom - flow_target).pow(2).mean()
    print(f"  t={t_val:.3f}: (1-t)={denom:.5f}, pred_flow_scale={pred_flow_scale:.1f}, "
          f"target_scale={flow_target_scale:.4f}, MSE={mse:.1f}")

# 4. Check video hidden scales
print("\n=== 4. Video Hidden Scales (Cosmos 2.0) ===")
wrapper = Cosmos2PredictWrapper(
    model_name="nvidia/Cosmos-Predict2-2B-Video2World",
    extract_layer=19,
)
wrapper = wrapper.to("cuda", dtype=torch.bfloat16).eval()

video_input = sample["video"].unsqueeze(0).cuda().bfloat16()  # (1, 2, 1, 3, 224, 224)
# Process one view
single_view = video_input[:, 0]  # (1, 1, 3, 224, 224)
with torch.no_grad():
    hidden = wrapper(single_view, prompts=["test"])
print(f"Video hidden: shape={hidden.shape}, dtype={hidden.dtype}")
print(f"  min={hidden.min():.4f}, max={hidden.max():.4f}")
print(f"  mean={hidden.mean():.4f}, std={hidden.std():.4f}")
print(f"  abs_mean={hidden.abs().mean():.4f}")

# Check if video_hidden_norm would help
from torch.nn import RMSNorm
norm = RMSNorm(hidden.shape[-1]).cuda()
normed = norm(hidden.float())
print(f"After RMSNorm: std={normed.std():.4f}, abs_mean={normed.abs().mean():.4f}")

# 5. Check action_normalizer impact
print("\n=== 5. Normalization Impact ===")
print("action_normalizer: NOT SET (None) — raw actions used directly")
print("joint_normalizer:  NOT SET (None) — raw joint states used directly")
print(f"video_hidden_norm: DEFINED but NOT CALLED in forward()")

# Check what normalized actions would look like
from mimic_video.data.normalization import load_stats
try:
    stats = load_stats(DATASET_PATH)
    print(f"Dataset stats available: {list(stats.keys())[:5]}...")
except Exception as e:
    print(f"No dataset stats: {e}")

print("\n=== Summary ===")
print("Likely causes of high loss:")
print("1. model_output_clean=True: divides by (1-t), t often near 1 → amplification up to 2000x")
print("2. No action normalization: raw actions ~0.01, noise ~1.0, imbalanced scales")
print("3. video_hidden_norm defined but never applied (line 384)")
print("4. No joint_state normalization")
