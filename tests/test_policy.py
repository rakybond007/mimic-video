"""
Tests for MimicVideoPolicy inference.
"""

import json
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch

from mimic_video.policy import MimicVideoPolicy


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def mock_policy():
    """Create a MimicVideoPolicy with a mock model."""
    mock_model = MagicMock()
    # model.sample() returns (1, action_chunk_len, max_action_dim)
    mock_model.sample.return_value = torch.rand(1, 16, 32)
    mock_model.to = MagicMock(return_value=mock_model)
    mock_model.eval = MagicMock(return_value=mock_model)
    mock_model.device = torch.device("cpu")

    stats = {
        "state.eef_pos_absolute": {
            "min": [-1.0, -1.0, 0.0],
            "max": [1.0, 1.0, 1.0],
            "mean": [0.0, 0.0, 0.5],
            "std": [0.5, 0.5, 0.3],
        },
        "state.eef_rot_absolute": {
            "min": [-3.14, -3.14, -3.14],
            "max": [3.14, 3.14, 3.14],
            "mean": [0.0, 0.0, 0.0],
            "std": [1.0, 1.0, 1.0],
        },
        "state.gripper_close": {
            "min": [0.0],
            "max": [1.0],
            "mean": [0.5],
            "std": [0.3],
        },
        "action.eef_pos_delta": {
            "min": [-0.5, -0.5, -0.5],
            "max": [0.5, 0.5, 0.5],
            "mean": [0.0, 0.0, 0.0],
            "std": [0.2, 0.2, 0.2],
        },
        "action.eef_rot_delta": {
            "min": [-0.5, -0.5, -0.5],
            "max": [0.5, 0.5, 0.5],
            "mean": [0.0, 0.0, 0.0],
            "std": [0.2, 0.2, 0.2],
        },
        "action.gripper_close": {
            "min": [0.0],
            "max": [1.0],
            "mean": [0.5],
            "std": [0.3],
        },
    }

    config = {
        "dim_action": 32,
        "dim_joint_state": 64,
        "denoising_steps": 16,
        "video_resolution": 224,
        "data_config_info": {
            "video_keys": ["video.front_view", "video.left_wrist_view"],
            "state_keys": [
                "state.eef_pos_absolute",
                "state.eef_rot_absolute",
                "state.gripper_close",
            ],
            "action_keys": [
                "action.eef_pos_delta",
                "action.eef_rot_delta",
                "action.gripper_close",
            ],
        },
    }

    return MimicVideoPolicy(model=mock_model, stats=stats, config=config, device="cpu")


@pytest.fixture
def sample_observation():
    """Create a sample LIBERO-style observation dict."""
    return {
        "video.image": np.random.randint(0, 256, (1, 256, 256, 3), dtype=np.uint8),
        "video.wrist_image": np.random.randint(0, 256, (1, 256, 256, 3), dtype=np.uint8),
        "state.x": np.array([[0.5]]),
        "state.y": np.array([[0.3]]),
        "state.z": np.array([[0.7]]),
        "state.roll": np.array([[0.1]]),
        "state.pitch": np.array([[-0.2]]),
        "state.yaw": np.array([[0.3]]),
        "state.gripper": np.array([[0.5, 0.5]]),
        "annotation.human.action.task_description": ["pick up the red cube"],
    }


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestMimicVideoPolicy:
    def test_get_action_returns_dict(self, mock_policy, sample_observation):
        result = mock_policy.get_action(sample_observation)
        assert isinstance(result, dict)
        # Should have action keys
        assert len(result) > 0

    def test_get_action_output_shapes(self, mock_policy, sample_observation):
        result = mock_policy.get_action(sample_observation)
        # Each action component should have horizon length (16)
        for key, val in result.items():
            assert key.startswith("action.")
            assert isinstance(val, np.ndarray)
            assert val.shape == (16,), f"Key {key} has shape {val.shape}, expected (16,)"

    def test_get_action_has_all_components(self, mock_policy, sample_observation):
        result = mock_policy.get_action(sample_observation)
        # Should have 7 components for LIBERO
        expected_keys = {"action.x", "action.y", "action.z",
                         "action.roll", "action.pitch", "action.yaw",
                         "action.gripper_close"}
        assert set(result.keys()) == expected_keys

    def test_model_sample_called_correctly(self, mock_policy, sample_observation):
        mock_policy.get_action(sample_observation)
        mock_policy.model.sample.assert_called_once()
        call_kwargs = mock_policy.model.sample.call_args[1]
        assert "steps" in call_kwargs
        assert call_kwargs["steps"] == 16
        assert "video" in call_kwargs
        assert "joint_state" in call_kwargs
        assert "prompts" in call_kwargs

    def test_preprocess_video_shape(self, mock_policy, sample_observation):
        video = mock_policy._preprocess_video(sample_observation)
        # Should be (1, V, T, C, H, W)
        assert video.ndim == 6
        assert video.shape[0] == 1  # batch
        assert video.shape[1] == 2  # 2 views
        assert video.shape[3] == 3  # channels
        assert video.shape[4] == 224  # height
        assert video.shape[5] == 224  # width

    def test_preprocess_state_shape(self, mock_policy, sample_observation):
        state = mock_policy._preprocess_state(sample_observation)
        assert state.shape == (1, 64)  # (1, max_state_dim)

    def test_get_prompts(self, mock_policy, sample_observation):
        prompts = mock_policy._get_prompts(sample_observation)
        assert isinstance(prompts, list)
        assert len(prompts) == 1
        assert prompts[0] == "pick up the red cube"

    def test_get_prompts_fallback(self, mock_policy):
        obs = {"some_other_key": "value"}
        prompts = mock_policy._get_prompts(obs)
        assert prompts == [""]

    def test_get_modality_config(self, mock_policy):
        config = mock_policy.get_modality_config()
        assert "video_keys" in config
        assert "state_keys" in config
        assert "action_keys" in config

    def test_postprocess_actions_denormalization(self, mock_policy):
        """Verify that actions are denormalized to the original scale."""
        # Create known normalized actions (all 0.5 in min-max → should be at midpoint)
        actions = torch.full((1, 16, 32), 0.5)
        result = mock_policy._postprocess_actions(actions)

        # For min_max with min=-0.5, max=0.5: denorm(0.5) = 0.5*(0.5-(-0.5)) + (-0.5) = 0.0
        for key in ["action.x", "action.y", "action.z"]:
            np.testing.assert_allclose(result[key], 0.0, atol=1e-5)

    def test_component_names_mapping(self, mock_policy):
        names = mock_policy._get_component_names("eef_pos_delta", 3)
        assert names == ["x", "y", "z"]

        names = mock_policy._get_component_names("eef_rot_delta", 3)
        assert names == ["roll", "pitch", "yaw"]

        names = mock_policy._get_component_names("gripper_close", 1)
        assert names == ["gripper"]
