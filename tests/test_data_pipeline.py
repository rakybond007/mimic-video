"""
Tests for the data pipeline: transforms, normalization, dataset, and data config.
"""

import json
import os
import shutil
import tempfile
from pathlib import Path

import numpy as np
import pytest
import torch

from mimic_video.data.data_config import LiberoDataConfig, load_data_config
from mimic_video.data.normalization import denormalize, normalize
from mimic_video.data.transforms import (
    Compose,
    StateActionNormalize,
    VideoColorJitter,
    VideoCrop,
    VideoResize,
    VideoToTensor,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def mini_dataset_path(tmp_path):
    """Create a minimal LeRobot-format dataset for testing."""
    dataset_dir = tmp_path / "mini_libero"
    meta_dir = dataset_dir / "meta"
    data_dir = dataset_dir / "data" / "chunk-000"
    video_dir = dataset_dir / "videos" / "chunk-000"
    meta_dir.mkdir(parents=True)
    data_dir.mkdir(parents=True)
    video_dir.mkdir(parents=True)

    # info.json
    info = {
        "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
        "video_path": "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4",
        "chunks_size": 1000,
        "features": {
            "video.front_view": {
                "shape": [3, 256, 256],
                "names": ["channel", "height", "width"],
                "info": {"video.channels": 3, "video.fps": 30},
            },
            "video.left_wrist_view": {
                "shape": [3, 256, 256],
                "names": ["channel", "height", "width"],
                "info": {"video.channels": 3, "video.fps": 30},
            },
        },
    }
    with open(meta_dir / "info.json", "w") as f:
        json.dump(info, f)

    # episodes.jsonl
    with open(meta_dir / "episodes.jsonl", "w") as f:
        f.write(json.dumps({"episode_index": 0, "length": 20}) + "\n")
        f.write(json.dumps({"episode_index": 1, "length": 15}) + "\n")

    # tasks.jsonl
    with open(meta_dir / "tasks.jsonl", "w") as f:
        f.write(json.dumps({"task_index": 0, "task": "pick up the red cup"}) + "\n")

    # modality.json
    modality_meta = {
        "state": {
            "eef_pos_absolute": {
                "original_key": "observation.state",
                "start": 0, "end": 3,
                "dtype": "float32", "absolute": True, "rotation_type": None,
            },
            "eef_rot_absolute": {
                "original_key": "observation.state",
                "start": 3, "end": 6,
                "dtype": "float32", "absolute": True, "rotation_type": None,
            },
            "gripper_close": {
                "original_key": "observation.state",
                "start": 6, "end": 7,
                "dtype": "float32", "absolute": True, "rotation_type": None,
            },
        },
        "action": {
            "eef_pos_delta": {
                "original_key": "action",
                "start": 0, "end": 3,
                "dtype": "float32", "absolute": False, "rotation_type": None,
            },
            "eef_rot_delta": {
                "original_key": "action",
                "start": 3, "end": 6,
                "dtype": "float32", "absolute": False, "rotation_type": None,
            },
            "gripper_close": {
                "original_key": "action",
                "start": 6, "end": 7,
                "dtype": "float32", "absolute": False, "rotation_type": None,
            },
        },
        "video": {
            "front_view": {"original_key": "video.front_view"},
            "left_wrist_view": {"original_key": "video.left_wrist_view"},
        },
    }
    with open(meta_dir / "modality.json", "w") as f:
        json.dump(modality_meta, f)

    # stats.json
    stats = {
        "observation.state": {
            "min": [-1.0] * 7, "max": [1.0] * 7,
            "mean": [0.0] * 7, "std": [0.5] * 7,
        },
        "action": {
            "min": [-0.5] * 7, "max": [0.5] * 7,
            "mean": [0.0] * 7, "std": [0.2] * 7,
        },
        "state.eef_pos_absolute": {
            "min": [-1.0] * 3, "max": [1.0] * 3,
            "mean": [0.0] * 3, "std": [0.5] * 3,
        },
        "state.eef_rot_absolute": {
            "min": [-1.0] * 3, "max": [1.0] * 3,
            "mean": [0.0] * 3, "std": [0.5] * 3,
        },
        "state.gripper_close": {
            "min": [0.0], "max": [1.0],
            "mean": [0.5], "std": [0.3],
        },
        "action.eef_pos_delta": {
            "min": [-0.5] * 3, "max": [0.5] * 3,
            "mean": [0.0] * 3, "std": [0.2] * 3,
        },
        "action.eef_rot_delta": {
            "min": [-0.5] * 3, "max": [0.5] * 3,
            "mean": [0.0] * 3, "std": [0.2] * 3,
        },
        "action.gripper_close": {
            "min": [0.0], "max": [1.0],
            "mean": [0.5], "std": [0.3],
        },
    }
    with open(meta_dir / "stats.json", "w") as f:
        json.dump(stats, f)

    # Create parquet files with synthetic data
    import pandas as pd

    for ep_id, ep_len in [(0, 20), (1, 15)]:
        records = []
        for step in range(ep_len):
            records.append({
                "observation.state": np.random.randn(7).astype(np.float32).tolist(),
                "action": np.random.randn(7).astype(np.float32).tolist(),
                "task_index": 0,
                "episode_index": ep_id,
                "frame_index": step,
                "index": step,
            })
        df = pd.DataFrame(records)
        parquet_path = data_dir / f"episode_{ep_id:06d}.parquet"
        df.to_parquet(parquet_path)

    return str(dataset_dir)


# ---------------------------------------------------------------------------
# Transform tests
# ---------------------------------------------------------------------------

class TestVideoTransforms:
    def test_video_to_tensor(self):
        video = np.random.randint(0, 256, (4, 64, 64, 3), dtype=np.uint8)
        t = VideoToTensor(apply_to=["video"])
        sample = t({"video": video})
        result = sample["video"]
        assert isinstance(result, torch.Tensor)
        assert result.shape == (4, 3, 64, 64)
        assert result.min() >= 0.0 and result.max() <= 1.0

    def test_video_crop(self):
        video = torch.rand(2, 3, 100, 100)
        t = VideoCrop(apply_to=["video"], scale=0.9, random=False)
        sample = t({"video": video})
        result = sample["video"]
        assert result.shape == (2, 3, 90, 90)

    def test_video_resize(self):
        video = torch.rand(2, 3, 100, 100)
        t = VideoResize(apply_to=["video"], height=224, width=224)
        sample = t({"video": video})
        result = sample["video"]
        assert result.shape == (2, 3, 224, 224)

    def test_video_color_jitter(self):
        video = torch.rand(2, 3, 64, 64)
        t = VideoColorJitter(apply_to=["video"])
        sample = t({"video": video})
        result = sample["video"]
        assert result.shape == (2, 3, 64, 64)
        assert result.min() >= 0.0 and result.max() <= 1.0

    def test_compose(self):
        video = np.random.randint(0, 256, (1, 100, 100, 3), dtype=np.uint8)
        pipeline = Compose([
            VideoToTensor(apply_to=["video"]),
            VideoCrop(apply_to=["video"], scale=0.95, random=False),
            VideoResize(apply_to=["video"], height=224, width=224),
        ])
        sample = pipeline({"video": video})
        result = sample["video"]
        assert result.shape == (1, 3, 224, 224)


class TestNormalization:
    def test_min_max_normalize_denormalize_roundtrip(self):
        stats = {"min": [0.0, -1.0], "max": [10.0, 1.0], "mean": [5.0, 0.0], "std": [3.0, 0.5]}
        value = np.array([5.0, 0.0])
        normalized = normalize(value, stats, mode="min_max")
        recovered = denormalize(normalized, stats, mode="min_max")
        np.testing.assert_allclose(recovered, value, atol=1e-6)

    def test_mean_std_normalize_denormalize_roundtrip(self):
        stats = {"min": [0.0], "max": [10.0], "mean": [5.0], "std": [2.0]}
        value = np.array([7.0])
        normalized = normalize(value, stats, mode="mean_std")
        recovered = denormalize(normalized, stats, mode="mean_std")
        np.testing.assert_allclose(recovered, value, atol=1e-6)

    def test_normalize_tensor(self):
        stats = {"min": [0.0, 0.0], "max": [1.0, 1.0], "mean": [0.5, 0.5], "std": [0.25, 0.25]}
        value = torch.tensor([0.5, 0.5])
        normalized = normalize(value, stats, mode="min_max")
        assert isinstance(normalized, torch.Tensor)
        recovered = denormalize(normalized, stats, mode="min_max")
        torch.testing.assert_close(recovered, value, atol=1e-6, rtol=1e-6)


class TestStateActionNormalize:
    def test_normalize_with_stats(self):
        stats = {
            "state.x": {"min": [0.0], "max": [10.0], "mean": [5.0], "std": [3.0]},
        }
        t = StateActionNormalize(
            apply_to=["state.x"],
            normalization_modes={"state.x": "min_max"},
        )
        sample = {"state.x": np.array([5.0])}
        result = t(sample, stats=stats)
        expected = 0.5  # (5 - 0) / (10 - 0) = 0.5
        assert abs(float(result["state.x"]) - expected) < 1e-6


# ---------------------------------------------------------------------------
# Data Config tests
# ---------------------------------------------------------------------------

class TestDataConfig:
    def test_libero_data_config_defaults(self):
        config = LiberoDataConfig()
        assert len(config.video_keys) == 2
        assert len(config.state_keys) == 3
        assert len(config.action_keys) == 3
        assert config.observation_indices == [0]
        assert len(config.action_indices) == 16

    def test_libero_data_config_multi_frame(self):
        config = LiberoDataConfig(num_frames=3)
        assert config.observation_indices == [-2, -1, 0]

    def test_modality_config(self):
        config = LiberoDataConfig()
        mc = config.modality_config()
        assert "video" in mc
        assert "state" in mc
        assert "action" in mc
        assert "language" in mc

    def test_transform_training(self):
        config = LiberoDataConfig()
        t = config.transform(training=True)
        assert isinstance(t, Compose)
        # Should include ColorJitter during training
        has_jitter = any(isinstance(x, VideoColorJitter) for x in t.transforms)
        assert has_jitter

    def test_transform_eval(self):
        config = LiberoDataConfig()
        t = config.transform(training=False)
        # Should NOT include ColorJitter during eval
        has_jitter = any(isinstance(x, VideoColorJitter) for x in t.transforms)
        assert not has_jitter

    def test_load_data_config(self):
        config = load_data_config("libero")
        assert isinstance(config, LiberoDataConfig)

    def test_load_data_config_unknown(self):
        with pytest.raises(ValueError, match="Unknown data config"):
            load_data_config("nonexistent")


# ---------------------------------------------------------------------------
# Dataset tests (requires synthetic dataset)
# ---------------------------------------------------------------------------

class TestDataset:
    def test_dataset_loads(self, mini_dataset_path):
        from mimic_video.data.dataset import LeRobotLiberoDataset

        ds = LeRobotLiberoDataset(
            dataset_path=mini_dataset_path,
            max_state_dim=64,
            max_action_dim=32,
        )
        assert len(ds) == 35  # 20 + 15 steps

    def test_dataset_getitem_shapes(self, mini_dataset_path):
        from mimic_video.data.dataset import LeRobotLiberoDataset

        ds = LeRobotLiberoDataset(
            dataset_path=mini_dataset_path,
            max_state_dim=64,
            max_action_dim=32,
        )
        sample = ds[0]

        # Check all required keys exist
        assert "video" in sample
        assert "actions" in sample
        assert "action_mask" in sample
        assert "joint_state" in sample
        assert "state_mask" in sample
        assert "prompt" in sample

        # Check action shapes
        assert sample["actions"].shape[0] == 16  # action horizon
        assert sample["actions"].shape[1] == 32  # max_action_dim
        assert sample["action_mask"].shape == (32,)
        assert sample["action_mask"].dtype == torch.bool

        # Check state shapes
        assert sample["joint_state"].shape[1] == 64  # max_state_dim
        assert sample["state_mask"].shape == (64,)
        assert sample["state_mask"].dtype == torch.bool

        # Check prompt
        assert isinstance(sample["prompt"], str)

    def test_action_mask_values(self, mini_dataset_path):
        """Verify mask has correct number of True values (7 real dims)."""
        from mimic_video.data.dataset import LeRobotLiberoDataset

        ds = LeRobotLiberoDataset(
            dataset_path=mini_dataset_path,
            max_state_dim=64,
            max_action_dim=32,
        )
        sample = ds[0]

        # LIBERO has 7 real action dims
        assert sample["action_mask"][:7].all()
        assert not sample["action_mask"][7:].any()

        # LIBERO has 7 real state dims
        assert sample["state_mask"][:7].all()
        assert not sample["state_mask"][7:].any()

    def test_dataset_not_found(self):
        from mimic_video.data.dataset import LeRobotLiberoDataset

        with pytest.raises(FileNotFoundError):
            LeRobotLiberoDataset(dataset_path="/nonexistent/path")
