"""G4: Real LIBERO dataset + tiny model training for 100 steps."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
from transformers import TrainingArguments

from mimic_video.mimic_video import MimicVideo
from mimic_video.cosmos_predict import CosmosPredictWrapper
from mimic_video.data.dataset import LeRobotLiberoDataset
from mimic_video.training.collator import MimicVideoDataCollator
from mimic_video.training.trainer import MimicVideoTrainer

DATASET_PATH = "/sjw_alinlab2/home/myungkyu/workspace/AlinVLA/.cache/huggingface/lerobot/kimtaey/libero_gr00t_delta"
OUTPUT_DIR = "/rlwrld3/home/hojin/vam_workspace/mimic-video/ckpt/g4_test"

# 1. Load real LIBERO dataset (video_resolution=64 for tiny VAE compatibility)
print("Loading dataset...")
dataset = LeRobotLiberoDataset(
    dataset_path=DATASET_PATH,
    num_frames=1,
    video_resolution=64,  # tiny VAE max_size=(4,16,16), spatial_compression=4
    max_state_dim=64,
    max_action_dim=32,
    training=True,
)
print(f"Dataset: {len(dataset)} steps")

# Verify first sample
sample = dataset[0]
print(f"  video: {sample['video'].shape}")
print(f"  actions: {sample['actions'].shape}")
print(f"  action_mask: {sample['action_mask'].shape} (True: {sample['action_mask'].sum().item()})")
print(f"  joint_state: {sample['joint_state'].shape}")
print(f"  prompt: {sample['prompt'][:60]}...")

# 2. Build tiny model
print("\nBuilding model...")
wrapper = CosmosPredictWrapper(random_weights=True, tiny=True)
model = MimicVideo(
    dim=64, video_predict_wrapper=wrapper,
    action_chunk_len=16, dim_action=32, dim_joint_state=64,
    num_video_viewpoints=2, depth=2, heads=2, dim_head=32,
)
total_params = sum(p.numel() for p in model.parameters())
print(f"Model params: {total_params:,}")

# 3. Training
print("\nStarting training (100 steps)...")
args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size=4,
    max_steps=100,
    logging_steps=10,
    save_steps=50,
    save_total_limit=2,
    report_to="none",
    remove_unused_columns=False,
    bf16=True,
    learning_rate=1e-4,
    warmup_ratio=0.05,
    lr_scheduler_type="cosine",
    seed=42,
)

trainer = MimicVideoTrainer(
    model=model,
    args=args,
    train_dataset=dataset,
    data_collator=MimicVideoDataCollator(),
    compute_dtype=torch.bfloat16,
)
trainer.train()

# 4. Check output
print(f"\nCheckpoints saved to: {OUTPUT_DIR}")
ckpt_dirs = sorted(os.listdir(OUTPUT_DIR)) if os.path.isdir(OUTPUT_DIR) else []
print(f"Contents: {ckpt_dirs}")

print("\nG4 PASSED")
