"""
Tests for training infrastructure: collator, trainer, and runner.
"""

import json
import os
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch
import torch.nn as nn

from mimic_video.training.collator import MimicVideoDataCollator


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_sample(batch_idx=0):
    """Create a single dataset sample matching the postprocess output format."""
    return {
        "video": torch.rand(2, 1, 3, 224, 224),     # (V, T, C, H, W)
        "actions": torch.rand(16, 32),                # (horizon, max_action_dim)
        "action_mask": torch.cat([
            torch.ones(7, dtype=torch.bool),
            torch.zeros(25, dtype=torch.bool),
        ]),
        "joint_state": torch.rand(1, 64),             # (T, max_state_dim)
        "state_mask": torch.cat([
            torch.ones(7, dtype=torch.bool),
            torch.zeros(57, dtype=torch.bool),
        ]),
        "prompt": f"pick up the object {batch_idx}",
    }


# ---------------------------------------------------------------------------
# Collator tests
# ---------------------------------------------------------------------------

class TestMimicVideoDataCollator:
    def test_collate_batch(self):
        collator = MimicVideoDataCollator()
        batch = [_make_sample(i) for i in range(4)]
        collated = collator(batch)

        assert collated["video"].shape == (4, 2, 1, 3, 224, 224)
        assert collated["actions"].shape == (4, 16, 32)
        assert collated["action_mask"].shape == (4, 32)
        assert collated["joint_state"].shape == (4, 1, 64)
        assert len(collated["prompts"]) == 4
        assert isinstance(collated["prompts"][0], str)

    def test_collate_single_sample(self):
        collator = MimicVideoDataCollator()
        batch = [_make_sample()]
        collated = collator(batch)

        assert collated["video"].shape[0] == 1
        assert collated["actions"].shape[0] == 1


# ---------------------------------------------------------------------------
# Trainer tests
# ---------------------------------------------------------------------------

class TestMimicVideoTrainer:
    def test_compute_loss_calls_model(self):
        """Test that compute_loss correctly unpacks inputs and calls model."""
        from mimic_video.training.trainer import MimicVideoTrainer

        # Create a mock model that returns a scalar loss
        mock_model = MagicMock()
        mock_model.return_value = torch.tensor(0.5, requires_grad=True)

        # Create minimal training args
        with tempfile.TemporaryDirectory() as tmpdir:
            from transformers import TrainingArguments

            args = TrainingArguments(
                output_dir=tmpdir,
                per_device_train_batch_size=1,
                max_steps=1,
                remove_unused_columns=False,
                report_to="none",
            )

            trainer = MimicVideoTrainer(
                model=mock_model,
                args=args,
                compute_dtype=torch.float32,
            )

            # Create a batch
            inputs = {
                "video": torch.rand(2, 2, 1, 3, 224, 224),
                "actions": torch.rand(2, 16, 32),
                "action_mask": torch.ones(2, 32, dtype=torch.bool),
                "joint_state": torch.rand(2, 1, 64),
                "prompts": ["task1", "task2"],
            }

            loss = trainer.compute_loss(mock_model, inputs)
            assert mock_model.called
            assert isinstance(loss, torch.Tensor)

    def test_compute_loss_with_return_outputs(self):
        """Test compute_loss with return_outputs=True."""
        from mimic_video.training.trainer import MimicVideoTrainer

        mock_model = MagicMock()
        mock_model.return_value = torch.tensor(0.5, requires_grad=True)

        with tempfile.TemporaryDirectory() as tmpdir:
            from transformers import TrainingArguments

            args = TrainingArguments(
                output_dir=tmpdir,
                per_device_train_batch_size=1,
                max_steps=1,
                remove_unused_columns=False,
                report_to="none",
            )

            trainer = MimicVideoTrainer(
                model=mock_model,
                args=args,
                compute_dtype=torch.float32,
            )

            inputs = {
                "video": torch.rand(2, 2, 1, 3, 224, 224),
                "actions": torch.rand(2, 16, 32),
                "action_mask": torch.ones(2, 32, dtype=torch.bool),
                "joint_state": torch.rand(2, 1, 64),
                "prompts": ["task1", "task2"],
            }

            loss, outputs = trainer.compute_loss(mock_model, inputs, return_outputs=True)
            assert isinstance(loss, torch.Tensor)
            assert "loss" in outputs


class TestMimicVideoSampler:
    def test_sampler_length(self):
        from mimic_video.training.trainer import MimicVideoSampler

        dataset = list(range(100))
        sampler = MimicVideoSampler(dataset, shuffle=False)
        assert len(sampler) == 100
        assert list(sampler) == list(range(100))

    def test_sampler_shuffle(self):
        from mimic_video.training.trainer import MimicVideoSampler

        dataset = list(range(100))
        sampler = MimicVideoSampler(dataset, shuffle=True, seed=42)
        indices = list(sampler)
        assert sorted(indices) == list(range(100))
        assert indices != list(range(100))  # Should be shuffled

    def test_sampler_epoch(self):
        from mimic_video.training.trainer import MimicVideoSampler

        dataset = list(range(100))
        sampler = MimicVideoSampler(dataset, shuffle=True, seed=42)

        epoch0 = list(sampler)
        sampler.set_epoch(1)
        epoch1 = list(sampler)
        assert epoch0 != epoch1  # Different epochs should give different orders
