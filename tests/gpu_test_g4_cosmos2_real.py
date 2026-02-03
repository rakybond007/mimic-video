"""G4-Cosmos2-Real: Real LIBERO dataset + real Cosmos 2.0 pretrained weights, 100 steps."""
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
OUTPUT_DIR = "/rlwrld3/home/hojin/vam_workspace/mimic-video/ckpt/g4_cosmos2_real"

# 1. Load real LIBERO dataset at 224x224 (real VAE: 8x spatial compression → 28x28 latent)
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

sample = dataset[0]
print(f"  video: {sample['video'].shape}")
print(f"  actions: {sample['actions'].shape}")
print(f"  action_mask: {sample['action_mask'].shape} (True: {sample['action_mask'].sum().item()})")
print(f"  joint_state: {sample['joint_state'].shape}")
print(f"  prompt: {sample['prompt'][:60]}...")

# 2. Build model with real Cosmos 2.0 pretrained weights
print("\nLoading Cosmos 2.0 pretrained weights (nvidia/Cosmos-Predict2-2B-Video2World)...")
wrapper = Cosmos2PredictWrapper(
    model_name="nvidia/Cosmos-Predict2-2B-Video2World",
    extract_layer=19,
)
print(f"  dim_latent: {wrapper.dim_latent}")
print(f"  vae_spatial_compression: {wrapper.vae_spatial_compression_ratio}")
print(f"  vae_temporal_compression: {wrapper.vae_temporal_compression_ratio}")
print(f"  transformer layers: {wrapper.transformer.config.num_layers}")
print(f"  concat_padding_mask: {wrapper.transformer.config.concat_padding_mask}")

print("\nBuilding MimicVideo...")
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

# Check GPU memory before training
if torch.cuda.is_available():
    print(f"\nGPU: {torch.cuda.get_device_name()}")
    print(f"GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

# 3. Training (100 steps, batch_size=2 for memory safety)
print("\nStarting training (100 steps)...")
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
)
trainer.train()

# 4. Check output
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

print("\nG4-Cosmos2-Real PASSED")
