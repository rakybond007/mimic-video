#!/usr/bin/env python3
"""
Self-contained test runner for mimic-video pipeline.
No pytest required — runs with plain Python.

Usage:
    python tests/run_tests.py
"""

import importlib
import os
import sys
import time
import traceback

# Ensure mimic-video package is importable without pip install -e
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import numpy as np
import torch

# ── Test infrastructure ──────────────────────────────────────────────────

RESULTS = []
CURRENT_SECTION = ""


def section(name):
    global CURRENT_SECTION
    CURRENT_SECTION = name
    print(f"\n{'='*70}")
    print(f"  {name}")
    print(f"{'='*70}")


def test(name):
    """Decorator that registers and runs a test function."""
    def decorator(fn):
        full_name = f"{CURRENT_SECTION}::{name}"
        try:
            fn()
            RESULTS.append((full_name, "PASS", ""))
            print(f"  PASS  {name}")
        except Exception as e:
            msg = traceback.format_exc()
            RESULTS.append((full_name, "FAIL", msg))
            print(f"  FAIL  {name}")
            print(f"        {e}")
        return fn
    return decorator


# ══════════════════════════════════════════════════════════════════════════
#  SECTION 1: Imports
# ══════════════════════════════════════════════════════════════════════════

section("1. Imports")


@test("mimic_video.MimicVideo")
def _():
    from mimic_video import MimicVideo
    assert MimicVideo is not None


@test("mimic_video.MimicVideoPolicy")
def _():
    from mimic_video import MimicVideoPolicy
    assert MimicVideoPolicy is not None


@test("mimic_video.data (transforms, config, normalization)")
def _():
    from mimic_video.data.transforms import (
        Compose, VideoToTensor, VideoCrop, VideoResize, VideoColorJitter, StateActionNormalize,
    )
    from mimic_video.data.data_config import BaseDataConfig, LiberoDataConfig, load_data_config
    from mimic_video.data.normalization import normalize, denormalize


@test("mimic_video.data.dataset")
def _():
    from mimic_video.data.dataset import BaseLeRobotDataset, LeRobotLiberoDataset


@test("mimic_video.training (collator, trainer, runner)")
def _():
    from mimic_video.training.collator import MimicVideoDataCollator
    from mimic_video.training.trainer import MimicVideoTrainer, MimicVideoSampler
    from mimic_video.training.runner import TrainRunner, CheckpointFormatCallback


@test("mimic_video.eval (libero_utils, server)")
def _():
    from mimic_video.eval.libero_utils import (
        quat2axisangle, normalize_gripper_action, TASK_SUITE_MAX_STEPS,
    )
    from mimic_video.eval.server import PolicyServer, PolicyClient, MsgSerializer


@test("mimic_video.policy")
def _():
    from mimic_video.policy import MimicVideoPolicy


# ══════════════════════════════════════════════════════════════════════════
#  SECTION 2: Video Transforms
# ══════════════════════════════════════════════════════════════════════════

section("2. Video Transforms")


@test("VideoToTensor: uint8 HWC → float32 CHW [0,1]")
def _():
    from mimic_video.data.transforms import VideoToTensor
    video = np.random.randint(0, 256, (4, 64, 64, 3), dtype=np.uint8)
    t = VideoToTensor(apply_to=["v"])
    result = t({"v": video})["v"]
    assert isinstance(result, torch.Tensor), f"Expected Tensor, got {type(result)}"
    assert result.shape == (4, 3, 64, 64), f"Shape {result.shape}"
    assert result.min() >= 0.0 and result.max() <= 1.0, f"Range [{result.min()}, {result.max()}]"


@test("VideoToTensor: single frame (H,W,C)")
def _():
    from mimic_video.data.transforms import VideoToTensor
    video = np.random.randint(0, 256, (64, 64, 3), dtype=np.uint8)
    t = VideoToTensor(apply_to=["v"])
    result = t({"v": video})["v"]
    assert result.shape == (3, 64, 64), f"Shape {result.shape}"


@test("VideoCrop: center crop 100→90")
def _():
    from mimic_video.data.transforms import VideoCrop
    video = torch.rand(2, 3, 100, 100)
    t = VideoCrop(apply_to=["v"], scale=0.9, random=False)
    result = t({"v": video})["v"]
    assert result.shape == (2, 3, 90, 90), f"Shape {result.shape}"


@test("VideoCrop: random crop size")
def _():
    from mimic_video.data.transforms import VideoCrop
    video = torch.rand(2, 3, 100, 100)
    t = VideoCrop(apply_to=["v"], scale=0.95, random=True)
    result = t({"v": video})["v"]
    assert result.shape == (2, 3, 95, 95), f"Shape {result.shape}"


@test("VideoResize: 90→224")
def _():
    from mimic_video.data.transforms import VideoResize
    video = torch.rand(2, 3, 90, 90)
    t = VideoResize(apply_to=["v"], height=224, width=224)
    result = t({"v": video})["v"]
    assert result.shape == (2, 3, 224, 224), f"Shape {result.shape}"


@test("VideoColorJitter: output stays in [0,1]")
def _():
    from mimic_video.data.transforms import VideoColorJitter
    video = torch.rand(4, 3, 64, 64)
    t = VideoColorJitter(apply_to=["v"], brightness=0.3, contrast=0.4, saturation=0.5, hue=0.08)
    result = t({"v": video})["v"]
    assert result.shape == (4, 3, 64, 64), f"Shape {result.shape}"
    assert result.min() >= 0.0, f"Min {result.min()}"
    assert result.max() <= 1.0, f"Max {result.max()}"


@test("Compose: full pipeline uint8 100×100 → float32 224×224")
def _():
    from mimic_video.data.transforms import Compose, VideoToTensor, VideoCrop, VideoResize
    video = np.random.randint(0, 256, (1, 100, 100, 3), dtype=np.uint8)
    pipeline = Compose([
        VideoToTensor(apply_to=["v"]),
        VideoCrop(apply_to=["v"], scale=0.95, random=False),
        VideoResize(apply_to=["v"], height=224, width=224),
    ])
    result = pipeline({"v": video})["v"]
    assert result.shape == (1, 3, 224, 224), f"Shape {result.shape}"


@test("Transform skips missing keys gracefully")
def _():
    from mimic_video.data.transforms import VideoToTensor
    t = VideoToTensor(apply_to=["missing_key"])
    result = t({"other": 123})
    assert result == {"other": 123}


# ══════════════════════════════════════════════════════════════════════════
#  SECTION 3: Normalization
# ══════════════════════════════════════════════════════════════════════════

section("3. Normalization")


@test("min_max normalize → denormalize roundtrip (numpy)")
def _():
    from mimic_video.data.normalization import normalize, denormalize
    stats = {"min": [0.0, -1.0], "max": [10.0, 1.0], "mean": [5.0, 0.0], "std": [3.0, 0.5]}
    val = np.array([5.0, 0.0])
    normed = normalize(val, stats, mode="min_max")
    recovered = denormalize(normed, stats, mode="min_max")
    np.testing.assert_allclose(recovered, val, atol=1e-6)


@test("mean_std normalize → denormalize roundtrip (numpy)")
def _():
    from mimic_video.data.normalization import normalize, denormalize
    stats = {"min": [0.0], "max": [10.0], "mean": [5.0], "std": [2.0]}
    val = np.array([7.0])
    normed = normalize(val, stats, mode="mean_std")
    recovered = denormalize(normed, stats, mode="mean_std")
    np.testing.assert_allclose(recovered, val, atol=1e-6)


@test("min_max normalize → denormalize roundtrip (Tensor)")
def _():
    from mimic_video.data.normalization import normalize, denormalize
    stats = {"min": [0.0, 0.0], "max": [1.0, 1.0], "mean": [0.5, 0.5], "std": [0.25, 0.25]}
    val = torch.tensor([0.5, 0.5])
    normed = normalize(val, stats, mode="min_max")
    assert isinstance(normed, torch.Tensor)
    recovered = denormalize(normed, stats, mode="min_max")
    torch.testing.assert_close(recovered, val, atol=1e-6, rtol=1e-6)


@test("min_max normalize: known values")
def _():
    from mimic_video.data.normalization import normalize
    stats = {"min": [0.0], "max": [10.0], "mean": [5.0], "std": [3.0]}
    assert abs(float(normalize(np.array([0.0]), stats, "min_max")) - 0.0) < 1e-8
    assert abs(float(normalize(np.array([10.0]), stats, "min_max")) - 1.0) < 1e-8
    assert abs(float(normalize(np.array([5.0]), stats, "min_max")) - 0.5) < 1e-8


@test("StateActionNormalize with stats dict")
def _():
    from mimic_video.data.transforms import StateActionNormalize
    stats = {"state.x": {"min": [0.0], "max": [10.0], "mean": [5.0], "std": [3.0]}}
    t = StateActionNormalize(apply_to=["state.x"], normalization_modes={"state.x": "min_max"})
    result = t({"state.x": np.array([5.0])}, stats=stats)
    assert abs(float(result["state.x"]) - 0.5) < 1e-6


@test("StateActionNormalize without stats → passthrough")
def _():
    from mimic_video.data.transforms import StateActionNormalize
    t = StateActionNormalize(apply_to=["state.x"], normalization_modes={"state.x": "min_max"})
    val = np.array([5.0])
    result = t({"state.x": val})
    np.testing.assert_array_equal(result["state.x"], val)


# ══════════════════════════════════════════════════════════════════════════
#  SECTION 4: Data Config
# ══════════════════════════════════════════════════════════════════════════

section("4. Data Config")


@test("LiberoDataConfig: default keys & indices")
def _():
    from mimic_video.data.data_config import LiberoDataConfig
    c = LiberoDataConfig()
    assert len(c.video_keys) == 2, f"video_keys: {c.video_keys}"
    assert len(c.state_keys) == 3, f"state_keys: {c.state_keys}"
    assert len(c.action_keys) == 3, f"action_keys: {c.action_keys}"
    assert c.observation_indices == [0]
    assert c.action_indices == list(range(16))


@test("LiberoDataConfig: num_frames=1 → [0]")
def _():
    from mimic_video.data.data_config import LiberoDataConfig
    c = LiberoDataConfig(num_frames=1)
    assert c.observation_indices == [0]


@test("LiberoDataConfig: num_frames=3 → [-2,-1,0]")
def _():
    from mimic_video.data.data_config import LiberoDataConfig
    c = LiberoDataConfig(num_frames=3)
    assert c.observation_indices == [-2, -1, 0], f"Got {c.observation_indices}"


@test("LiberoDataConfig: num_frames=5 → [-4,-3,-2,-1,0]")
def _():
    from mimic_video.data.data_config import LiberoDataConfig
    c = LiberoDataConfig(num_frames=5)
    assert c.observation_indices == [-4, -3, -2, -1, 0]


@test("modality_config(): 4 modalities with correct keys")
def _():
    from mimic_video.data.data_config import LiberoDataConfig
    mc = LiberoDataConfig().modality_config()
    assert set(mc.keys()) == {"video", "state", "action", "language"}
    assert mc["video"].modality_keys == ["video.front_view", "video.left_wrist_view"]
    assert mc["action"].delta_indices == list(range(16))


@test("transform(training=True): includes ColorJitter")
def _():
    from mimic_video.data.data_config import LiberoDataConfig
    from mimic_video.data.transforms import VideoColorJitter
    t = LiberoDataConfig().transform(training=True)
    assert any(isinstance(x, VideoColorJitter) for x in t.transforms)


@test("transform(training=False): no ColorJitter")
def _():
    from mimic_video.data.data_config import LiberoDataConfig
    from mimic_video.data.transforms import VideoColorJitter
    t = LiberoDataConfig().transform(training=False)
    assert not any(isinstance(x, VideoColorJitter) for x in t.transforms)


@test("load_data_config('libero')")
def _():
    from mimic_video.data.data_config import load_data_config, LiberoDataConfig
    c = load_data_config("libero")
    assert isinstance(c, LiberoDataConfig)


@test("load_data_config('nonexistent') → ValueError")
def _():
    from mimic_video.data.data_config import load_data_config
    try:
        load_data_config("nonexistent")
        assert False, "Should have raised ValueError"
    except ValueError:
        pass


# ══════════════════════════════════════════════════════════════════════════
#  SECTION 5: Collator
# ══════════════════════════════════════════════════════════════════════════

section("5. Collator")


def _make_sample(idx=0):
    return {
        "video": torch.rand(2, 1, 3, 224, 224),
        "actions": torch.rand(16, 32),
        "action_mask": torch.cat([torch.ones(7, dtype=torch.bool), torch.zeros(25, dtype=torch.bool)]),
        "joint_state": torch.rand(1, 64),
        "state_mask": torch.cat([torch.ones(7, dtype=torch.bool), torch.zeros(57, dtype=torch.bool)]),
        "prompt": f"task {idx}",
    }


@test("collate batch of 4: correct shapes")
def _():
    from mimic_video.training.collator import MimicVideoDataCollator
    c = MimicVideoDataCollator()
    out = c([_make_sample(i) for i in range(4)])
    assert out["video"].shape == (4, 2, 1, 3, 224, 224), f"video: {out['video'].shape}"
    assert out["actions"].shape == (4, 16, 32), f"actions: {out['actions'].shape}"
    assert out["action_mask"].shape == (4, 32), f"action_mask: {out['action_mask'].shape}"
    assert out["joint_state"].shape == (4, 1, 64), f"joint_state: {out['joint_state'].shape}"
    assert len(out["prompts"]) == 4
    assert out["prompts"][2] == "task 2"


@test("collate single sample")
def _():
    from mimic_video.training.collator import MimicVideoDataCollator
    c = MimicVideoDataCollator()
    out = c([_make_sample()])
    assert out["video"].shape[0] == 1
    assert out["actions"].shape[0] == 1


# ══════════════════════════════════════════════════════════════════════════
#  SECTION 6: Sampler
# ══════════════════════════════════════════════════════════════════════════

section("6. Sampler")


@test("length matches dataset")
def _():
    from mimic_video.training.trainer import MimicVideoSampler
    s = MimicVideoSampler(list(range(100)), shuffle=False)
    assert len(s) == 100
    assert list(s) == list(range(100))


@test("shuffle produces permutation")
def _():
    from mimic_video.training.trainer import MimicVideoSampler
    s = MimicVideoSampler(list(range(100)), shuffle=True, seed=42)
    indices = list(s)
    assert sorted(indices) == list(range(100))
    assert indices != list(range(100)), "Should be shuffled"


@test("different epochs → different orders")
def _():
    from mimic_video.training.trainer import MimicVideoSampler
    s = MimicVideoSampler(list(range(100)), shuffle=True, seed=42)
    epoch0 = list(s)
    s.set_epoch(1)
    epoch1 = list(s)
    assert epoch0 != epoch1, "Different epochs should produce different orders"


@test("same seed+epoch → reproducible")
def _():
    from mimic_video.training.trainer import MimicVideoSampler
    s1 = MimicVideoSampler(list(range(50)), shuffle=True, seed=99)
    s2 = MimicVideoSampler(list(range(50)), shuffle=True, seed=99)
    assert list(s1) == list(s2)


# ══════════════════════════════════════════════════════════════════════════
#  SECTION 7: Trainer compute_loss
# ══════════════════════════════════════════════════════════════════════════

section("7. Trainer compute_loss")


class _StubModel(torch.nn.Module):
    """Minimal nn.Module that records call args for testing compute_loss."""
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(1, 1)  # need at least one param
        self.last_kwargs = {}
        self.call_count = 0
        self._loss_val = 0.5

    def forward(self, **kwargs):
        self.last_kwargs = kwargs
        self.call_count += 1
        return torch.tensor(self._loss_val, requires_grad=True)


@test("compute_loss unpacks inputs and returns scalar")
def _():
    from mimic_video.training.trainer import MimicVideoTrainer
    from transformers import TrainingArguments
    import tempfile

    stub = _StubModel()
    with tempfile.TemporaryDirectory() as d:
        args = TrainingArguments(
            output_dir=d, per_device_train_batch_size=1, max_steps=1,
            remove_unused_columns=False, report_to="none",
        )
        trainer = MimicVideoTrainer(model=stub, args=args, compute_dtype=torch.float32)

        inputs = {
            "video": torch.rand(2, 2, 1, 3, 224, 224),
            "actions": torch.rand(2, 16, 32),
            "action_mask": torch.ones(2, 32, dtype=torch.bool),
            "joint_state": torch.rand(2, 1, 64),
            "prompts": ["a", "b"],
        }
        loss = trainer.compute_loss(stub, inputs)
        assert stub.call_count == 1, "model not called"
        assert loss.shape == (), f"Expected scalar, got {loss.shape}"


@test("compute_loss with return_outputs=True")
def _():
    from mimic_video.training.trainer import MimicVideoTrainer
    from transformers import TrainingArguments
    import tempfile

    stub = _StubModel()
    stub._loss_val = 1.0
    with tempfile.TemporaryDirectory() as d:
        args = TrainingArguments(
            output_dir=d, per_device_train_batch_size=1, max_steps=1,
            remove_unused_columns=False, report_to="none",
        )
        trainer = MimicVideoTrainer(model=stub, args=args, compute_dtype=torch.float32)

        inputs = {
            "video": torch.rand(1, 2, 1, 3, 224, 224),
            "actions": torch.rand(1, 16, 32),
            "action_mask": torch.ones(1, 32, dtype=torch.bool),
            "joint_state": torch.rand(1, 1, 64),
            "prompts": ["a"],
        }
        loss, outputs = trainer.compute_loss(stub, inputs, return_outputs=True)
        assert "loss" in outputs


@test("compute_loss: joint_state (B,T,D) → takes last timestep")
def _():
    from mimic_video.training.trainer import MimicVideoTrainer
    from transformers import TrainingArguments
    import tempfile

    stub = _StubModel()
    with tempfile.TemporaryDirectory() as d:
        args = TrainingArguments(
            output_dir=d, per_device_train_batch_size=1, max_steps=1,
            remove_unused_columns=False, report_to="none",
        )
        trainer = MimicVideoTrainer(model=stub, args=args, compute_dtype=torch.float32)

        inputs = {
            "video": torch.rand(2, 2, 1, 3, 224, 224),
            "actions": torch.rand(2, 16, 32),
            "action_mask": torch.ones(2, 32, dtype=torch.bool),
            "joint_state": torch.rand(2, 3, 64),  # T=3
            "prompts": ["a", "b"],
        }
        trainer.compute_loss(stub, inputs)
        # joint_state passed to model should be (B, D) not (B, T, D)
        assert stub.last_kwargs["joint_state"].shape == (2, 64), \
            f"Got {stub.last_kwargs['joint_state'].shape}"


# ══════════════════════════════════════════════════════════════════════════
#  SECTION 8: Policy (mock model)
# ══════════════════════════════════════════════════════════════════════════

section("8. Policy")

_POLICY_STATS = {
    "state.eef_pos_absolute": {"min": [-1.0]*3, "max": [1.0]*3, "mean": [0.0]*3, "std": [0.5]*3},
    "state.eef_rot_absolute": {"min": [-3.14]*3, "max": [3.14]*3, "mean": [0.0]*3, "std": [1.0]*3},
    "state.gripper_close": {"min": [0.0], "max": [1.0], "mean": [0.5], "std": [0.3]},
    "action.eef_pos_delta": {"min": [-0.5]*3, "max": [0.5]*3, "mean": [0.0]*3, "std": [0.2]*3},
    "action.eef_rot_delta": {"min": [-0.5]*3, "max": [0.5]*3, "mean": [0.0]*3, "std": [0.2]*3},
    "action.gripper_close": {"min": [0.0], "max": [1.0], "mean": [0.5], "std": [0.3]},
}
_POLICY_CONFIG = {
    "dim_action": 32, "dim_joint_state": 64, "denoising_steps": 16, "video_resolution": 224,
    "data_config_info": {
        "video_keys": ["video.front_view", "video.left_wrist_view"],
        "state_keys": ["state.eef_pos_absolute", "state.eef_rot_absolute", "state.gripper_close"],
        "action_keys": ["action.eef_pos_delta", "action.eef_rot_delta", "action.gripper_close"],
    },
}


def _make_mock_policy():
    from unittest.mock import MagicMock
    from mimic_video.policy import MimicVideoPolicy

    mock_model = MagicMock()
    mock_model.sample.return_value = torch.rand(1, 16, 32)
    mock_model.to = MagicMock(return_value=mock_model)
    mock_model.eval = MagicMock(return_value=mock_model)
    return MimicVideoPolicy(model=mock_model, stats=_POLICY_STATS, config=_POLICY_CONFIG, device="cpu")


def _make_obs():
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


@test("get_action returns dict")
def _():
    p = _make_mock_policy()
    result = p.get_action(_make_obs())
    assert isinstance(result, dict), f"Type: {type(result)}"
    assert len(result) > 0


@test("get_action: all 7 action components present")
def _():
    p = _make_mock_policy()
    result = p.get_action(_make_obs())
    expected = {"action.x", "action.y", "action.z", "action.roll", "action.pitch", "action.yaw", "action.gripper_close"}
    assert set(result.keys()) == expected, f"Keys: {set(result.keys())}"


@test("get_action: each component has shape (16,)")
def _():
    p = _make_mock_policy()
    result = p.get_action(_make_obs())
    for key, val in result.items():
        assert isinstance(val, np.ndarray), f"{key}: type={type(val)}"
        assert val.shape == (16,), f"{key}: shape={val.shape}"


@test("model.sample called with correct args")
def _():
    p = _make_mock_policy()
    p.get_action(_make_obs())
    kw = p.model.sample.call_args[1]
    assert kw["steps"] == 16
    assert "video" in kw
    assert "joint_state" in kw
    assert "prompts" in kw


@test("_preprocess_video: output shape (1, 2, T, 3, 224, 224)")
def _():
    p = _make_mock_policy()
    video = p._preprocess_video(_make_obs())
    assert video.ndim == 6
    assert video.shape[0] == 1  # batch
    assert video.shape[1] == 2  # views
    assert video.shape[3] == 3  # channels
    assert video.shape[4] == 224
    assert video.shape[5] == 224


@test("_preprocess_state: output shape (1, 64)")
def _():
    p = _make_mock_policy()
    state = p._preprocess_state(_make_obs())
    assert state.shape == (1, 64), f"Shape: {state.shape}"


@test("_get_prompts: extracts task description")
def _():
    p = _make_mock_policy()
    prompts = p._get_prompts(_make_obs())
    assert prompts == ["pick up the red cube"]


@test("_get_prompts: fallback to empty string")
def _():
    p = _make_mock_policy()
    assert p._get_prompts({"unrelated": 1}) == [""]


@test("_postprocess_actions: denormalization correctness")
def _():
    p = _make_mock_policy()
    # 0.5 in min_max with [-0.5, 0.5] → denorm = 0.5*(0.5-(-0.5))+(-0.5) = 0.0
    actions = torch.full((1, 16, 32), 0.5)
    result = p._postprocess_actions(actions)
    for key in ["action.x", "action.y", "action.z"]:
        np.testing.assert_allclose(result[key], 0.0, atol=1e-5, err_msg=f"{key}")


@test("_get_component_names mappings")
def _():
    p = _make_mock_policy()
    assert p._get_component_names("eef_pos_delta", 3) == ["x", "y", "z"]
    assert p._get_component_names("eef_rot_delta", 3) == ["roll", "pitch", "yaw"]
    assert p._get_component_names("gripper_close", 1) == ["gripper"]


@test("get_modality_config returns expected keys")
def _():
    p = _make_mock_policy()
    mc = p.get_modality_config()
    assert "video_keys" in mc
    assert "state_keys" in mc
    assert "action_keys" in mc


# ══════════════════════════════════════════════════════════════════════════
#  SECTION 9: Eval Utils
# ══════════════════════════════════════════════════════════════════════════

section("9. Eval Utils")


@test("quat2axisangle: identity quaternion → zero")
def _():
    from mimic_video.eval.libero_utils import quat2axisangle
    result = quat2axisangle(np.array([0.0, 0.0, 0.0, 1.0]))
    np.testing.assert_allclose(result, np.zeros(3), atol=1e-8)


@test("quat2axisangle: 90° around z-axis")
def _():
    from mimic_video.eval.libero_utils import quat2axisangle
    import math
    angle = math.pi / 2
    q = np.array([0.0, 0.0, math.sin(angle/2), math.cos(angle/2)])
    result = quat2axisangle(q)
    expected = np.array([0.0, 0.0, angle])
    np.testing.assert_allclose(result, expected, atol=1e-6)


@test("normalize_gripper_action: 0 → +1, 1 → -1")
def _():
    from mimic_video.eval.libero_utils import normalize_gripper_action
    a = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.0])
    result = normalize_gripper_action(a.copy(), binarize=True)
    assert result[-1] == 1.0, f"gripper=0 → expected +1, got {result[-1]}"

    a2 = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 1.0])
    result2 = normalize_gripper_action(a2.copy(), binarize=True)
    assert result2[-1] == -1.0, f"gripper=1 → expected -1, got {result2[-1]}"


@test("TASK_SUITE_MAX_STEPS has all suites")
def _():
    from mimic_video.eval.libero_utils import TASK_SUITE_MAX_STEPS
    expected = {"libero_spatial", "libero_object", "libero_goal", "libero_10", "libero_90"}
    assert set(TASK_SUITE_MAX_STEPS.keys()) == expected


# ══════════════════════════════════════════════════════════════════════════
#  SECTION 10: Server serialization
# ══════════════════════════════════════════════════════════════════════════

section("10. Server serialization")

try:
    import msgpack
    HAS_MSGPACK = True
except ImportError:
    HAS_MSGPACK = False


@test("MsgSerializer numpy roundtrip")
def _():
    if not HAS_MSGPACK:
        print("        (skipped — msgpack not installed)")
        return
    from mimic_video.eval.server import MsgSerializer
    data = {"arr": np.array([1.0, 2.0, 3.0]), "text": "hello"}
    encoded = MsgSerializer.to_bytes(data)
    decoded = MsgSerializer.from_bytes(encoded)
    np.testing.assert_array_equal(decoded["arr"], data["arr"])
    assert decoded["text"] == "hello"


@test("MsgSerializer empty dict")
def _():
    if not HAS_MSGPACK:
        print("        (skipped — msgpack not installed)")
        return
    from mimic_video.eval.server import MsgSerializer
    encoded = MsgSerializer.to_bytes({})
    decoded = MsgSerializer.from_bytes(encoded)
    assert decoded == {}


# ══════════════════════════════════════════════════════════════════════════
#  SUMMARY
# ══════════════════════════════════════════════════════════════════════════

print(f"\n{'='*70}")
print(f"  SUMMARY")
print(f"{'='*70}\n")

passed = sum(1 for _, s, _ in RESULTS if s == "PASS")
failed = sum(1 for _, s, _ in RESULTS if s == "FAIL")
total = len(RESULTS)

for name, status, msg in RESULTS:
    marker = "✓" if status == "PASS" else "✗"
    print(f"  {marker} {name}")

print(f"\n  {passed}/{total} passed, {failed} failed\n")

if failed > 0:
    print("  FAILED TESTS:\n")
    for name, status, msg in RESULTS:
        if status == "FAIL":
            print(f"  --- {name} ---")
            print(msg)
    sys.exit(1)
else:
    print("  All tests passed!")
    sys.exit(0)
