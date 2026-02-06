"""
MimicVideoTrainer: HuggingFace Trainer subclass for MimicVideo training.

Follows GR00T's DualBrainTrainer pattern from gr00t/experiment/trainer.py.
Key difference: MimicVideo.forward() uses keyword args and returns a loss tensor
directly, so compute_loss() uses model(**inputs) instead of model(inputs).
"""

import json
import os
from typing import Optional

import numpy as np
import torch
import transformers
from safetensors.torch import save_file
from torch.utils.data import Dataset, Sampler

try:
    from transformers.trainer import (
        ALL_LAYERNORM_LAYERS,
        TRAINER_STATE_NAME,
        TrainerState,
        get_last_checkpoint,
        get_parameter_names,
        is_sagemaker_mp_enabled,
    )
except ImportError:
    from transformers.pytorch_utils import ALL_LAYERNORM_LAYERS
    from transformers.trainer import (
        TRAINER_STATE_NAME,
        TrainerState,
        get_last_checkpoint,
        get_parameter_names,
        is_sagemaker_mp_enabled,
    )


class MimicVideoSampler(Sampler):
    """Epoch-aware sampler with shuffle. Same as GR00T's BaseSampler."""

    def __init__(self, data_source: Dataset, shuffle: bool = False, seed: int = 0):
        self.data_source = data_source
        self.shuffle = shuffle
        self.seed = seed
        self.epoch = 0

    def __iter__(self):
        if self.shuffle:
            g = torch.Generator()
            g.manual_seed(self.seed + self.epoch)
            return iter(torch.randperm(len(self.data_source), generator=g).tolist())
        return iter(range(len(self.data_source)))

    def set_epoch(self, epoch):
        self.epoch = epoch
        if hasattr(self.data_source, "set_epoch"):
            self.data_source.set_epoch(epoch)

    def __len__(self):
        return len(self.data_source)


class MimicVideoTrainer(transformers.Trainer):
    """
    Trainer subclass for MimicVideo.

    Follows the same pattern as GR00T's DualBrainTrainer:
    - Custom compute_loss that calls model with the right interface
    - Custom optimizer with decay/no-decay param groups
    - Custom sampler with epoch awareness
    - Custom save_model for model.pt format
    """

    def __init__(self, **kwargs):
        self.compute_dtype = kwargs.pop("compute_dtype", torch.bfloat16)
        self.freeze_video_backbone = kwargs.pop("freeze_video_backbone", False)
        super().__init__(**kwargs)
        if self.freeze_video_backbone:
            self._freeze_video_backbone()
        # Allowlist numpy globals for safe RNG state unpickling in PyTorch 2.1+
        torch.serialization.add_safe_globals(
            [np.core.multiarray._reconstruct, np.ndarray, np.dtype, np.dtypes.UInt32DType]
        )

    def _freeze_video_backbone(self):
        """Freeze video_predict_wrapper parameters so only the action head trains."""
        wrapper = getattr(self.model, "video_predict_wrapper", None)
        if wrapper is None:
            return
        wrapper.requires_grad_(False)
        frozen = sum(p.numel() for p in wrapper.parameters())
        trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print(f"Froze video backbone: {frozen:,} params frozen, {trainable:,} trainable")

    def _get_train_sampler(self, train_dataset: Optional[Dataset] = None):
        return MimicVideoSampler(train_dataset, shuffle=True, seed=self.args.seed)

    def _get_eval_sampler(self, eval_dataset: Optional[Dataset] = None):
        return MimicVideoSampler(eval_dataset, shuffle=False)

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        """
        MimicVideo.forward() takes keyword args and returns loss tensor directly.
        We unpack the collated batch into the model's expected kwargs.
        """
        # Extract fields from collated batch
        video = inputs["video"]  # (B, V, T, C, H, W)
        actions = inputs["actions"]  # (B, horizon, max_action_dim)
        action_mask = inputs["action_mask"]  # (B, max_action_dim)
        joint_state = inputs["joint_state"]  # (B, T, max_state_dim)
        prompts = inputs.get("prompts")  # list[str] or None
        future_video = inputs.get("future_video")  # (B, V, T_future, C, H, W) or None

        # MimicVideo.forward() expects joint_state as (B, D) - take the last timestep (current state)
        if joint_state.ndim == 3:
            joint_state_input = joint_state[:, -1, :]  # (B, max_state_dim)
        else:
            joint_state_input = joint_state

        loss = model(
            actions=actions,
            joint_state=joint_state_input,
            action_mask=action_mask,
            video=video,
            future_video=future_video,
            prompts=prompts,
            no_grad_video_model_forward=self.freeze_video_backbone,
        )

        if return_outputs:
            return loss, {"loss": loss}
        return loss

    def create_optimizer(self):
        """
        Setup optimizer with decay/no-decay parameter groups.
        Identical to DualBrainTrainer.create_optimizer().
        """
        if is_sagemaker_mp_enabled():
            return super().create_optimizer()

        opt_model = self.model

        if self.optimizer is None:
            decay_parameters = get_parameter_names(opt_model, ALL_LAYERNORM_LAYERS)
            decay_parameters = [name for name in decay_parameters if "bias" not in name]
            optimizer_grouped_parameters = [
                {
                    "params": [
                        p
                        for n, p in opt_model.named_parameters()
                        if (n in decay_parameters and p.requires_grad)
                    ],
                    "weight_decay": self.args.weight_decay,
                },
                {
                    "params": [
                        p
                        for n, p in opt_model.named_parameters()
                        if (n not in decay_parameters and p.requires_grad)
                    ],
                    "weight_decay": 0.0,
                },
            ]

            optimizer_cls, optimizer_kwargs = transformers.Trainer.get_optimizer_cls_and_kwargs(
                self.args
            )
            self.optimizer = optimizer_cls(optimizer_grouped_parameters, **optimizer_kwargs)

        return self.optimizer

    def save_model(self, output_dir: Optional[str] = None, _internal_call: bool = False):
        """Save model in sharded safetensors format with index.json."""
        output_dir = output_dir or self.args.output_dir
        os.makedirs(output_dir, exist_ok=True)

        if self.is_deepspeed_enabled:
            state_dict = self.accelerator.get_state_dict(self.deepspeed)
        else:
            state_dict = self.model.state_dict()

        if not self.args.should_save:
            return

        # Move to CPU contiguous for safetensors
        state_dict = {k: v.cpu().contiguous() for k, v in state_dict.items()}

        total_size = sum(v.numel() * v.element_size() for v in state_dict.values())
        total_params = sum(v.numel() for v in state_dict.values())
        max_shard_size = 4 * 1024 * 1024 * 1024  # 4GB per shard

        # Build shards
        shards = []  # list of {key: tensor}
        current_shard = {}
        current_size = 0

        for key, tensor in state_dict.items():
            tensor_size = tensor.numel() * tensor.element_size()
            if current_size + tensor_size > max_shard_size and current_shard:
                shards.append(current_shard)
                current_shard = {}
                current_size = 0
            current_shard[key] = tensor
            current_size += tensor_size

        if current_shard:
            shards.append(current_shard)

        num_shards = len(shards)
        weight_map = {}

        for i, shard_dict in enumerate(shards, 1):
            shard_name = f"model-{i:05d}-of-{num_shards:05d}.safetensors"
            save_file(shard_dict, os.path.join(output_dir, shard_name))
            for key in shard_dict:
                weight_map[key] = shard_name

        # Write index
        index = {
            "metadata": {
                "total_size": total_size,
                "total_parameters": total_params,
            },
            "weight_map": weight_map,
        }
        with open(os.path.join(output_dir, "model.safetensors.index.json"), "w") as f:
            json.dump(index, f, indent=2)

    def train(
        self,
        resume_from_checkpoint=None,
        trial=None,
        ignore_keys_for_eval=None,
        **kwargs,
    ):
        """Handle resume logic. Same as DualBrainTrainer.train()."""
        if resume_from_checkpoint is False:
            resume_from_checkpoint = None

        if isinstance(resume_from_checkpoint, bool) and resume_from_checkpoint:
            resume_from_checkpoint = get_last_checkpoint(self.args.output_dir)
            if resume_from_checkpoint is None:
                raise ValueError(
                    f"No valid checkpoint found in output directory ({self.args.output_dir})"
                )

        if resume_from_checkpoint is not None:
            self.state = TrainerState.load_from_json(
                os.path.join(resume_from_checkpoint, TRAINER_STATE_NAME)
            )
        return super().train(resume_from_checkpoint, trial, ignore_keys_for_eval, **kwargs)
