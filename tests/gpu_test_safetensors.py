"""Quick test: verify safetensors checkpoint format."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import shutil
import torch
from torch.utils.data import Dataset
from transformers import TrainingArguments

from mimic_video.mimic_video import MimicVideo
from mimic_video.cosmos_predict import CosmosPredictWrapper
from mimic_video.training.collator import MimicVideoDataCollator
from mimic_video.training.trainer import MimicVideoTrainer

OUTPUT_DIR = "/tmp/test_safetensors_ckpt"
if os.path.exists(OUTPUT_DIR):
    shutil.rmtree(OUTPUT_DIR)


class DummyDataset(Dataset):
    def __len__(self):
        return 20

    def __getitem__(self, idx):
        return {
            "video": torch.rand(2, 1, 3, 64, 64),
            "actions": torch.rand(16, 32),
            "action_mask": torch.cat([torch.ones(7, dtype=torch.bool), torch.zeros(25, dtype=torch.bool)]),
            "joint_state": torch.rand(1, 64),
            "prompt": "test task",
        }


wrapper = CosmosPredictWrapper(random_weights=True, tiny=True)
model = MimicVideo(
    dim=64, video_predict_wrapper=wrapper,
    action_chunk_len=16, dim_action=32, dim_joint_state=64,
    num_video_viewpoints=2, depth=2, heads=2, dim_head=32,
)

args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size=2,
    max_steps=3,
    logging_steps=1,
    save_steps=2,
    report_to="none",
    remove_unused_columns=False,
    bf16=True,
)

trainer = MimicVideoTrainer(
    model=model, args=args,
    train_dataset=DummyDataset(),
    data_collator=MimicVideoDataCollator(),
    compute_dtype=torch.bfloat16,
)
trainer.train()

# Check output structure
print("\n=== Checkpoint structure ===")
for root, dirs, files in os.walk(OUTPUT_DIR):
    level = root.replace(OUTPUT_DIR, "").count(os.sep)
    indent = "  " * level
    basename = os.path.basename(root)
    print(f"{indent}{basename}/")
    subindent = "  " * (level + 1)
    for f in sorted(files):
        size = os.path.getsize(os.path.join(root, f))
        print(f"{subindent}{f}  ({size:,} bytes)")

# Verify safetensors files exist
import json
ckpt_dir = os.path.join(OUTPUT_DIR, "checkpoint-2")
index_path = os.path.join(ckpt_dir, "model.safetensors.index.json")
assert os.path.exists(index_path), f"Missing {index_path}"

with open(index_path) as f:
    index = json.load(f)
print(f"\n=== Index metadata ===")
print(f"total_parameters: {index['metadata']['total_parameters']:,}")
print(f"total_size: {index['metadata']['total_size']:,} bytes")
print(f"num weight keys: {len(index['weight_map'])}")
shard_files = sorted(set(index['weight_map'].values()))
print(f"shard files: {shard_files}")

for sf in shard_files:
    assert os.path.exists(os.path.join(ckpt_dir, sf)), f"Missing shard: {sf}"

# Verify we can reload the weights
from safetensors.torch import load_file
loaded = {}
for sf in shard_files:
    loaded.update(load_file(os.path.join(ckpt_dir, sf)))
print(f"\nLoaded {len(loaded)} parameters from safetensors")
assert set(loaded.keys()) == set(model.state_dict().keys()), "Key mismatch!"

# Clean up
shutil.rmtree(OUTPUT_DIR)
print("\nSafetensors format test PASSED")
