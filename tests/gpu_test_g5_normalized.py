"""G5: Direct velocity prediction (model_output_clean=False), matching AlinVLA's approach.

Key fix: MimicVideo defaults to model_output_clean=True (x0 prediction), which divides
by (1-t) and inflates loss 100-1000x when t→1. AlinVLA predicts velocity directly.
Setting model_output_clean=False makes both approaches equivalent.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
from transformers import TrainingArguments

from mimic_video.mimic_video import MimicVideo
from mimic_video.cosmos_predict import Cosmos2PredictWrapper
from mimic_video.data.dataset import LeRobotLiberoDataset
from mimic_video.training.collator import MimicVideoDataCollator
from mimic_video.training.trainer import MimicVideoTrainer

DATASET_PATH = "/sjw_alinlab2/home/myungkyu/workspace/AlinVLA/.cache/huggingface/lerobot/kimtaey/libero_gr00t_delta"
OUTPUT_DIR = "/rlwrld3/home/hojin/vam_workspace/mimic-video/ckpt/g5_velocity"

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

# 2. Build model: model_output_clean=False (direct velocity prediction, like AlinVLA)
print("\nLoading Cosmos 2.0 pretrained (nvidia/Cosmos-Predict2-2B-Video2World)...")
wrapper = Cosmos2PredictWrapper(
    model_name="nvidia/Cosmos-Predict2-2B-Video2World",
    extract_layer=19,
)
print(f"  dim_latent: {wrapper.dim_latent}")

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
    model_output_clean=False,  # KEY: direct velocity prediction (like AlinVLA)
)

total_params = sum(p.numel() for p in model.parameters())
print(f"Total params: {total_params:,}")
print(f"model_output_clean: {model.model_output_clean}")

# 3. Training with frozen video backbone
print("\nStarting training (100 steps, video backbone FROZEN, velocity prediction)...")
args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size=2,
    gradient_accumulation_steps=2,
    max_steps=100,
    logging_steps=10,
    save_steps=50,
    save_total_limit=2,
    report_to="none",
    remove_unused_columns=False,
    bf16=True,
    learning_rate=1e-4,
    adam_beta1=0.95,
    adam_beta2=0.999,
    weight_decay=1e-5,
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
    freeze_video_backbone=True,
)
trainer.train()

# 4. Results
print(f"\nCheckpoints saved to: {OUTPUT_DIR}")
ckpt_dirs = sorted(os.listdir(OUTPUT_DIR)) if os.path.isdir(OUTPUT_DIR) else []
print(f"Contents: {ckpt_dirs}")

for d in ckpt_dirs:
    full = os.path.join(OUTPUT_DIR, d)
    if os.path.isdir(full):
        files = os.listdir(full)
        safetensors_files = [f for f in files if f.endswith('.safetensors')]
        index_file = 'model.safetensors.index.json' in files
        print(f"  {d}/: {len(safetensors_files)} safetensors, index={index_file}")

if torch.cuda.is_available():
    print(f"\nPeak GPU memory: {torch.cuda.max_memory_allocated() / 1e9:.2f} GB")

print("\nG5 PASSED")
