"""
TrainRunner: orchestrates MimicVideo training setup.

Following GR00T's TrainRunner from gr00t/experiment/runner.py.
"""

import json
import os
import shutil
from pathlib import Path

import torch
from transformers import TrainingArguments, set_seed
from transformers import TrainerCallback

from mimic_video.training.collator import MimicVideoDataCollator
from mimic_video.training.trainer import MimicVideoTrainer


class CheckpointFormatCallback(TrainerCallback):
    """Copy experiment_cfg directory into each checkpoint."""

    def __init__(self, run_name: str, exp_cfg_dir: Path):
        self.run_name = run_name
        self.exp_cfg_dir = exp_cfg_dir

    def on_save(self, args, state, control, **kwargs):
        checkpoint_dir = Path(args.output_dir) / f"checkpoint-{state.global_step}"
        if checkpoint_dir.exists() and self.exp_cfg_dir.exists():
            dest = checkpoint_dir / "experiment_cfg"
            if not dest.exists():
                shutil.copytree(str(self.exp_cfg_dir), str(dest))


class TrainRunner:
    """
    Orchestrates the full training setup, mirroring gr00t/experiment/runner.py.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        training_args: TrainingArguments,
        train_dataset,
        resume_from_checkpoint: bool | str | None = None,
        freeze_video_backbone: bool = True,
    ):
        self.training_args = training_args
        self.output_dir = Path(training_args.output_dir)
        self.exp_cfg_dir = self.output_dir / "experiment_cfg"
        self.exp_cfg_dir.mkdir(parents=True, exist_ok=True)
        self.resume_from_checkpoint = resume_from_checkpoint
        self.train_dataset = train_dataset

        # Set run name
        training_args.run_name = (
            training_args.output_dir.split("/")[-1]
            if training_args.run_name is None
            else training_args.run_name
        )
        rank = int(os.environ.get("RANK", 0))
        if rank == 0:
            print(f"Run name: {training_args.run_name}")

        # Data collator
        data_collator = MimicVideoDataCollator()

        # Compute dtype
        compute_dtype = torch.bfloat16 if training_args.bf16 else torch.float32
        set_seed(training_args.seed)

        # Create trainer
        self.trainer = MimicVideoTrainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            data_collator=data_collator,
            compute_dtype=compute_dtype,
            freeze_video_backbone=freeze_video_backbone,
        )

        # Add checkpoint callback
        ckpt_callback = CheckpointFormatCallback(
            run_name=training_args.run_name,
            exp_cfg_dir=self.exp_cfg_dir,
        )
        self.trainer.add_callback(ckpt_callback)

        # Save metadata
        if rank == 0:
            self._save_metadata()

        # Set up reporting
        self._setup_reporting()

        # Log info (only on main process to avoid duplicate output)
        if rank == 0:
            print(
                f"Train dataset length: {len(train_dataset)}\n"
                f"GPU memory before training: {torch.cuda.memory_allocated() / 1024**3:.2f} GB"
                if torch.cuda.is_available()
                else f"Train dataset length: {len(train_dataset)}"
            )

    def _save_metadata(self):
        """Save dataset stats and config to experiment_cfg/metadata.json."""
        metadata = {}
        metadata_path = self.exp_cfg_dir / "metadata.json"

        if metadata_path.exists():
            with open(metadata_path, "r") as f:
                metadata = json.load(f)

        # Save dataset stats if available
        if hasattr(self.train_dataset, "stats"):
            metadata["dataset_stats"] = self.train_dataset.stats

        # Save data config info
        if hasattr(self.train_dataset, "data_config"):
            config = self.train_dataset.data_config
            metadata["data_config"] = {
                "video_keys": config.video_keys,
                "state_keys": config.state_keys,
                "action_keys": config.action_keys,
                "language_keys": config.language_keys,
                "observation_indices": config.observation_indices,
                "action_indices": config.action_indices,
            }

        if hasattr(self.train_dataset, "max_state_dim"):
            metadata["max_state_dim"] = self.train_dataset.max_state_dim
        if hasattr(self.train_dataset, "max_action_dim"):
            metadata["max_action_dim"] = self.train_dataset.max_action_dim

        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=4)

    def _setup_reporting(self):
        """Configure reporting (tensorboard/wandb)."""
        report_to = self.training_args.report_to
        # report_to can be str or list from TrainingArguments
        if isinstance(report_to, list):
            report_to = report_to[0] if report_to else "tensorboard"
        if report_to == "wandb":
            if "WANDB_PROJECT" not in os.environ:
                os.environ["WANDB_PROJECT"] = "mimic-video-training"
            os.environ["WANDB_DIR"] = self.training_args.output_dir
            self.training_args.report_to = ["wandb"]
        elif report_to == "none":
            self.training_args.report_to = ["none"]
        else:
            tensorboard_dir = self.output_dir / "runs"
            tensorboard_dir.mkdir(parents=True, exist_ok=True)
            if int(os.environ.get("RANK", 0)) == 0:
                print(f"TensorBoard logs will be saved to: {tensorboard_dir}")
            self.training_args.report_to = ["tensorboard"]

    def train(self):
        """Run training."""
        self.trainer.train(resume_from_checkpoint=self.resume_from_checkpoint)
        self.trainer.save_state()
        self.trainer.save_model(self.training_args.output_dir)
