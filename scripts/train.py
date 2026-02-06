#!/usr/bin/env python3
"""
Main training entry point for MimicVideo fine-tuning.

Usage:
    python scripts/train.py --config configs/libero.yaml --dataset_path /path/to/data

    # With CLI overrides:
    python scripts/train.py --config configs/libero.yaml \
        --dataset_path /path/to/data \
        --learning_rate 5e-5 \
        --max_steps 10000 \
        --lora_rank 8

Reference: Isaac-GR00T-AlinVLA/scripts/gr00t_finetune.py
"""

import argparse
import json
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path

import torch
import torch.distributed as dist
import yaml
from transformers import TrainingArguments

from mimic_video.data.data_config import load_data_config
from mimic_video.data.dataset import LeRobotLiberoDataset
from mimic_video.data.normalization import compute_normalizer_stats, load_stats
from mimic_video.training.runner import TrainRunner


def _is_main_process():
    return int(os.environ.get("LOCAL_RANK", 0)) == 0


def _world_size():
    return int(os.environ.get("WORLD_SIZE", 1))


def load_config(config_path: str, cli_overrides: dict) -> dict:
    """Load YAML config and merge CLI overrides."""
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    # CLI overrides take precedence
    for key, value in cli_overrides.items():
        if value is not None:
            config[key] = value
    return config


def build_video_wrapper(config: dict):
    """Build Cosmos video wrapper (separated for multi-GPU serialization)."""
    from mimic_video.cosmos_predict import CosmosPredictWrapper, Cosmos2PredictWrapper

    video_model = config.get("video_model", "nvidia/Cosmos-Predict2-2B-Video2World")
    if "Predict2" in video_model or "2.5" in video_model:
        return Cosmos2PredictWrapper(
            model_name=video_model,
            extract_layer=config.get("extract_layer", 19),
        )
    else:
        return CosmosPredictWrapper(
            model_name=video_model,
            extract_layer=config.get("extract_layer", 19),
        )


def build_model(config: dict, dataset=None, video_wrapper=None) -> torch.nn.Module:
    """Build MimicVideo model with CosmosPredictWrapper."""
    from mimic_video.mimic_video import MimicVideo

    if video_wrapper is None:
        video_wrapper = build_video_wrapper(config)

    # Get action/joint normalization stats from dataset's stats.json if available
    # Falls back to computing from samples if stats.json not found
    action_mean_std = None
    joint_mean_std = None
    if dataset is not None:
        stats = getattr(dataset, '_stats', None) or getattr(dataset, 'stats', {})
        use_precomputed = False
        dim_action = config.get("dim_action", 32)
        dim_state = config.get("dim_joint_state", 64)

        if stats:
            # Try to build normalizer stats from dataset's stats.json
            # LeRobot stats format: {"action": {"mean": [...], "std": [...]}, "observation.state": {...}}
            try:
                action_stats = stats.get("action", {})
                state_stats = stats.get("observation.state", {})

                if action_stats and state_stats:
                    action_means = action_stats.get("mean", [])
                    action_stds = action_stats.get("std", [])
                    state_means = state_stats.get("mean", [])
                    state_stds = state_stats.get("std", [])

                    if action_means and state_means:
                        # Pad to max dims
                        action_mean = torch.zeros(dim_action)
                        action_std = torch.ones(dim_action)
                        action_mean[:len(action_means)] = torch.tensor(action_means, dtype=torch.float32)
                        action_std[:len(action_stds)] = torch.tensor(action_stds, dtype=torch.float32).clamp(min=1e-6)

                        state_mean = torch.zeros(dim_state)
                        state_std = torch.ones(dim_state)
                        state_mean[:len(state_means)] = torch.tensor(state_means, dtype=torch.float32)
                        state_std[:len(state_stds)] = torch.tensor(state_stds, dtype=torch.float32).clamp(min=1e-6)

                        action_mean_std = torch.stack([action_mean, action_std])
                        joint_mean_std = torch.stack([state_mean, state_std])
                        use_precomputed = True

                        if _is_main_process():
                            real_a = len(action_means)
                            real_s = len(state_means)
                            print(f"Using precomputed stats from dataset's stats.json")
                            print(f"  action: {real_a} dims, state: {real_s} dims")
                            print(f"  action_mean_std: mean=[{action_mean_std[0,:real_a].min():.4f}, {action_mean_std[0,:real_a].max():.4f}], "
                                  f"std=[{action_mean_std[1,:real_a].min():.4f}, {action_mean_std[1,:real_a].max():.4f}]")
            except (KeyError, TypeError) as e:
                if _is_main_process():
                    print(f"Could not use precomputed stats: {e}")

        if not use_precomputed:
            if _is_main_process():
                print("Computing action/joint normalization stats from dataset samples...")
            action_mean_std, joint_mean_std = compute_normalizer_stats(dataset, num_samples=5000)
            if _is_main_process():
                real_a = action_mean_std[0][action_mean_std[1] < 0.999].numel()
                print(f"  action_mean_std: shape={action_mean_std.shape}, "
                      f"real dims mean range=[{action_mean_std[0,:real_a].min():.4f}, {action_mean_std[0,:real_a].max():.4f}], "
                      f"std range=[{action_mean_std[1,:real_a].min():.4f}, {action_mean_std[1,:real_a].max():.4f}]")

    # Build MimicVideo model
    model = MimicVideo(
        dim=config.get("dim", 512),
        video_predict_wrapper=video_wrapper,
        action_chunk_len=config.get("action_chunk_len", 16),
        dim_action=config.get("dim_action", 32),
        dim_joint_state=config.get("dim_joint_state", 64),
        proprio_mask_prob=config.get("proprio_mask_prob", 0.1),
        depth=config.get("depth", 8),
        dim_head=config.get("dim_head", 64),
        heads=config.get("heads", 8),
        num_video_viewpoints=config.get("num_video_viewpoints", 2),
        model_output_clean=config.get("model_output_clean", False),
        inject_language_tokens=config.get("inject_language_tokens", False),
        action_mean_std=action_mean_std,
        joint_mean_std=joint_mean_std,
    )

    return model


def apply_lora(model: torch.nn.Module, config: dict) -> torch.nn.Module:
    """Apply LoRA to model if configured."""
    lora_rank = config.get("lora_rank", 0)
    if lora_rank <= 0:
        return model

    from peft import LoraConfig, get_peft_model

    lora_config = LoraConfig(
        r=lora_rank,
        lora_alpha=config.get("lora_alpha", 16),
        lora_dropout=config.get("lora_dropout", 0.1),
        target_modules=["to_q", "to_k", "to_v", "to_out"],
        bias="none",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    return model


def build_training_args(config: dict) -> TrainingArguments:
    """Build HuggingFace TrainingArguments from config."""
    return TrainingArguments(
        output_dir=config.get("output_dir", "./checkpoints/libero"),
        learning_rate=float(config.get("learning_rate", 1e-4)),
        adam_beta1=float(config.get("adam_beta1", 0.95)),
        adam_beta2=float(config.get("adam_beta2", 0.999)),
        weight_decay=float(config.get("weight_decay", 1e-5)),
        warmup_ratio=float(config.get("warmup_ratio", 0.05)),
        per_device_train_batch_size=int(config.get("per_device_train_batch_size", 4)),
        gradient_accumulation_steps=int(config.get("gradient_accumulation_steps", 8)),
        max_steps=int(config.get("max_steps", 50000)),
        save_steps=int(config.get("save_steps", 5000)),
        save_total_limit=int(config.get("save_total_limit", 5)),
        logging_steps=int(config.get("logging_steps", 10)),
        bf16=config.get("bf16", True),
        tf32=config.get("tf32", True),
        lr_scheduler_type=config.get("lr_scheduler_type", "cosine"),
        report_to=config.get("report_to", "tensorboard"),
        seed=int(config.get("seed", 42)),
        remove_unused_columns=False,  # We handle our own collation
    )


def main():
    # ---- Distributed setup ----
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    world_size = _world_size()
    is_main = _is_main_process()

    if world_size > 1:
        if not dist.is_initialized():
            dist.init_process_group(backend="nccl")
        torch.cuda.set_device(local_rank)

    parser = argparse.ArgumentParser(description="Train MimicVideo on LIBERO")
    parser.add_argument("--config", type=str, default="configs/libero.yaml", help="Path to YAML config")
    parser.add_argument("--dataset_path", type=str, default=None, help="Override dataset path")
    parser.add_argument("--output_dir", type=str, default=None, help="Override output directory")
    parser.add_argument("--learning_rate", type=float, default=None)
    parser.add_argument("--max_steps", type=int, default=None)
    parser.add_argument("--per_device_train_batch_size", type=int, default=None)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=None)
    parser.add_argument("--lora_rank", type=int, default=None)
    parser.add_argument("--resume_from_checkpoint", action="store_true", default=None)
    parser.add_argument("--num_frames", type=int, default=None)
    parser.add_argument("--video_resolution", type=int, default=None)
    parser.add_argument("--freeze_video_backbone", action="store_true", default=None,
                        help="Freeze Cosmos video backbone, train only MimicVideo head")
    parser.add_argument("--run_name", type=str, default=None, help="Wandb/TB run name")
    parser.add_argument("--report_to", type=str, default=None, help="Reporting: wandb, tensorboard, none")
    parser.add_argument("--save_steps", type=int, default=None)
    parser.add_argument("--logging_steps", type=int, default=None)
    parser.add_argument("--inject_language_tokens", action="store_true", default=None,
                        help="Inject T5 language tokens into action head cross-attention")
    parser.add_argument("--num_future_frames", type=int, default=None,
                        help="Number of future video frames for Algorithm 2 (clean past, noised future)")
    args = parser.parse_args()

    # Build config
    cli_overrides = {k: v for k, v in vars(args).items() if k != "config" and v is not None}
    config = load_config(args.config, cli_overrides)

    # Validate required fields
    dataset_path = config.get("dataset_path", "")
    if not dataset_path:
        print("ERROR: --dataset_path is required")
        sys.exit(1)

    if is_main:
        print(f"Config: {json.dumps(config, indent=2, default=str)}")

    # 1. Create dataset
    if is_main:
        print("Loading dataset...")

    # Import data config for future_frames support
    from mimic_video.data.data_config import LiberoDataConfig

    data_config = LiberoDataConfig(
        num_frames=config.get("num_frames", 1),
        video_resolution=config.get("video_resolution", 224),
        num_future_frames=config.get("num_future_frames", 0),
    )

    dataset = LeRobotLiberoDataset(
        dataset_path=dataset_path,
        data_config=data_config,
        max_state_dim=config.get("dim_joint_state", 64),
        max_action_dim=config.get("dim_action", 32),
        training=True,
    )
    if is_main:
        print(f"Dataset loaded: {len(dataset)} steps")
        if config.get("num_future_frames", 0) > 0:
            print(f"  Using Algorithm 2 with {config.get('num_future_frames')} future frames")

    # 2. Build model
    # For multi-GPU: rank 0 builds video wrapper first (triggers HF Hub downloads),
    # then barrier, then other ranks build from cache.
    if is_main:
        print("Building model...")

    if world_size > 1:
        if is_main:
            video_wrapper = build_video_wrapper(config)
        dist.barrier()
        if not is_main:
            video_wrapper = build_video_wrapper(config)
    else:
        video_wrapper = build_video_wrapper(config)

    model = build_model(config, dataset=dataset, video_wrapper=video_wrapper)

    # 3. Apply LoRA if configured
    model = apply_lora(model, config)

    # 4. Build training args
    training_args = build_training_args(config)

    # Override run_name if provided
    if config.get("run_name"):
        training_args.run_name = config["run_name"]

    # 5. Create runner and train
    freeze_backbone = config.get("freeze_video_backbone", True)
    runner = TrainRunner(
        model=model,
        training_args=training_args,
        train_dataset=dataset,
        resume_from_checkpoint=config.get("resume_from_checkpoint", False),
        freeze_video_backbone=freeze_backbone,
    )
    runner.train()
    if is_main:
        print("Training complete!")


if __name__ == "__main__":
    main()
