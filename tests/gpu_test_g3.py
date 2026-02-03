"""G3: 합성 데이터로 training loop 5스텝"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
from torch.utils.data import Dataset
from mimic_video.mimic_video import MimicVideo
from mimic_video.cosmos_predict import CosmosPredictWrapper
from mimic_video.training.collator import MimicVideoDataCollator
from mimic_video.training.trainer import MimicVideoTrainer
from transformers import TrainingArguments


class DummyDataset(Dataset):
    def __len__(self):
        return 100

    def __getitem__(self, idx):
        return {
            "video": torch.rand(2, 1, 3, 64, 64),  # tiny VAE expects <=64x64
            "actions": torch.rand(16, 32),
            "action_mask": torch.cat([
                torch.ones(7, dtype=torch.bool),
                torch.zeros(25, dtype=torch.bool),
            ]),
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
    output_dir="/tmp/test_training_g3",
    per_device_train_batch_size=2,
    max_steps=5,
    logging_steps=1,
    save_steps=999,
    report_to="none",
    remove_unused_columns=False,
    bf16=True,
)

trainer = MimicVideoTrainer(
    model=model,
    args=args,
    train_dataset=DummyDataset(),
    data_collator=MimicVideoDataCollator(),
    compute_dtype=torch.bfloat16,
)
trainer.train()
print("G3 PASSED")
