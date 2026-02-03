"""G4 with real Cosmos-Predict2.5-2B + real LIBERO dataset, 100 steps."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
from transformers import TrainingArguments

from mimic_video.cosmos_predict import Cosmos2_5PredictWrapper
from mimic_video.mimic_video import MimicVideo
from mimic_video.data.dataset import LeRobotLiberoDataset
from mimic_video.training.collator import MimicVideoDataCollator
from mimic_video.training.trainer import MimicVideoTrainer

DATASET_PATH = "/sjw_alinlab2/home/myungkyu/workspace/AlinVLA/.cache/huggingface/lerobot/kimtaey/libero_gr00t_delta"
OUTPUT_DIR = "/rlwrld3/home/hojin/vam_workspace/mimic-video/ckpt/g4_cosmos25"

# 1. Load dataset
print("Loading dataset...")
dataset = LeRobotLiberoDataset(
    dataset_path=DATASET_PATH,
    num_frames=1,
    video_resolution=224,
    max_state_dim=64,
    max_action_dim=32,
    training=True,
)
print(f"Dataset: {len(dataset)} steps")

# 2. Build model with Cosmos 2.5
print("\nLoading Cosmos-Predict2.5-2B (this may download ~5GB on first run)...")
wrapper = Cosmos2_5PredictWrapper(
    model_name="nvidia/Cosmos-Predict2.5-2B",
    extract_layer=19,
)
print(f"Cosmos loaded. Latent dim: {wrapper.dim_latent}")

model = MimicVideo(
    dim=512,
    video_predict_wrapper=wrapper,
    action_chunk_len=16,
    dim_action=32,
    dim_joint_state=64,
    num_video_viewpoints=2,
    depth=8,
    heads=8,
    dim_head=64,
)
total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Total params: {total_params:,}")
print(f"Trainable params: {trainable_params:,}")

# 3. Training
print("\nStarting training (100 steps)...")
args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size=2,
    gradient_accumulation_steps=1,
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

# 4. Verify checkpoint format
print(f"\nCheckpoints saved to: {OUTPUT_DIR}")
for item in sorted(os.listdir(OUTPUT_DIR)):
    full = os.path.join(OUTPUT_DIR, item)
    if os.path.isdir(full):
        files = os.listdir(full)
        st_files = [f for f in files if f.endswith(".safetensors")]
        idx_files = [f for f in files if f.endswith(".index.json")]
        print(f"  {item}/  safetensors={len(st_files)} index={len(idx_files)}")
    else:
        print(f"  {item}  ({os.path.getsize(full):,} bytes)")

print("\nG4 (Cosmos 2.5) PASSED")
