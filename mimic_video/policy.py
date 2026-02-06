"""
MimicVideoPolicy: wraps MimicVideo for inference.

Handles: raw observations -> normalize -> model.sample() -> denormalize -> actions.

Following AlinVLA/GR00T pattern:
- StateActionNormalize transform normalizes state/action to [-1, 1] using min_max
- Same transform is applied at both training and inference
- transform.unapply() denormalizes action output back to original scale

Reference: gr00t/model/policy.py (GR00TPolicy)
"""

import json
from pathlib import Path
from typing import Any, Optional

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor

from mimic_video.data.transforms import StateActionNormalize


class MimicVideoPolicy:
    """
    Wraps MimicVideo for inference with AlinVLA-style normalization.

    Preprocessing:
      - video: uint8 HWC -> float32 CHW [0,1], resize to video_resolution
      - state: normalize to [-1, 1] using min_max, then pad to max_state_dim
    Post-processing:
      - model.sample() returns normalized actions in [-1, 1]
      - transform.unapply() converts back to original scale
      - extract first real_action_dim dims from the 32-dim output
    """

    # LIBERO produces 8-dim state: eef_pos(3) + axis_angle(3) + gripper_qpos(2)
    LIBERO_REAL_STATE_DIM = 8
    LIBERO_REAL_ACTION_DIM = 7

    def __init__(
        self,
        model,
        config: dict,
        state_action_transform: Optional[StateActionNormalize] = None,
        device: str | torch.device = "cuda",
    ):
        self.model = model
        self.config = config
        self.device = torch.device(device) if isinstance(device, str) else device
        self.model.to(self.device)
        self.model.eval()

        # Extract config values
        self.video_resolution = config.get("video_resolution", 224)
        self.max_action_dim = config.get("dim_action", 32)
        self.max_state_dim = config.get("dim_joint_state", 64)
        self.denoising_steps = config.get("denoising_steps", 16)
        self.real_action_dim = config.get("real_action_dim", self.LIBERO_REAL_ACTION_DIM)
        self.real_state_dim = config.get("real_state_dim", self.LIBERO_REAL_STATE_DIM)

        # StateActionNormalize transform for input normalization and output denormalization
        self.state_action_transform = state_action_transform

    def get_action(self, observation: dict[str, Any]) -> dict[str, np.ndarray]:
        """
        Get actions from observations.

        Args:
            observation: Dict with:
                "video.front_view": (T, H, W, 3) uint8 or "video.image"
                "video.left_wrist_view": (T, H, W, 3) uint8 or "video.wrist_image"
                "state.eef_pos_absolute": (3,) float
                "state.eef_rot_absolute": (3,) float
                "state.gripper_close": (N,) float
                "annotation.human.action.task_description": [str]

        Returns:
            Dict with:
                "action.eef_pos_delta": (horizon, 3)
                "action.eef_rot_delta": (horizon, 3)
                "action.gripper_close": (horizon, 1)
        """
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            # 1. Preprocess video (no normalization needed, just resize)
            video = self._preprocess_video(observation)  # (1, V, T, C, H, W)

            # 2. Preprocess state with normalization
            joint_state = self._preprocess_state(observation)  # (1, max_state_dim)

            # 3. Get prompt
            prompts = self._get_prompts(observation)

            # 4. Sample actions (model outputs normalized values in [-1, 1])
            actions = self.model.sample(
                steps=self.denoising_steps,
                batch_size=1,
                video=video,
                joint_state=joint_state,
                prompts=prompts,
                disable_progress_bar=True,
            )  # (1, action_chunk_len, max_action_dim)

            # 5. Denormalize and extract real action dims
            return self._postprocess_actions(actions)

    def _preprocess_video(self, observation: dict) -> Tensor:
        """Preprocess video observations into model input format."""
        views = []

        # Try both naming conventions
        for obs_key in ["video.front_view", "video.image",
                        "video.left_wrist_view", "video.wrist_image"]:
            if obs_key not in observation:
                continue
            frames = observation[obs_key]
            if isinstance(frames, np.ndarray):
                frames = torch.from_numpy(frames.copy()).float()

            # frames shape: (T, H, W, C) or (H, W, C)
            if frames.ndim == 3:
                frames = frames.unsqueeze(0)  # (1, H, W, C)

            # HWC -> CHW and normalize to [0,1]
            if frames.shape[-1] == 3:  # HWC format
                frames = frames.permute(0, 3, 1, 2)  # (T, C, H, W)

            if frames.max() > 1.0:
                frames = frames / 255.0

            # Resize to model resolution
            frames = F.interpolate(
                frames,
                size=(self.video_resolution, self.video_resolution),
                mode="bilinear",
                align_corners=False,
            )

            views.append(frames)  # (T, C, H, W)

        if not views:
            raise ValueError("No video keys found in observation")

        # Stack views: (V, T, C, H, W) -> add batch: (1, V, T, C, H, W)
        video = torch.stack(views, dim=0).unsqueeze(0).to(self.device)
        return video

    def _preprocess_state(self, observation: dict) -> Tensor:
        """
        Preprocess state observations with normalization and padding.

        Uses StateActionNormalize to normalize to [-1, 1] (AlinVLA style).
        """
        # Build state dict for individual keys
        state_dict = {}

        # Try grouped keys first (server/client mode sends these)
        if "state.eef_pos_absolute" in observation:
            state_dict["state.eef_pos_absolute"] = torch.tensor(
                np.atleast_1d(observation["state.eef_pos_absolute"]).flatten(),
                dtype=torch.float32
            )
            state_dict["state.eef_rot_absolute"] = torch.tensor(
                np.atleast_1d(observation["state.eef_rot_absolute"]).flatten(),
                dtype=torch.float32
            )
            state_dict["state.gripper_close"] = torch.tensor(
                np.atleast_1d(observation["state.gripper_close"]).flatten(),
                dtype=torch.float32
            )
        # Try individual component keys (from _build_observation_dict)
        elif "state.x" in observation:
            pos = np.array([
                float(np.atleast_1d(observation["state.x"]).flatten()[-1]),
                float(np.atleast_1d(observation["state.y"]).flatten()[-1]),
                float(np.atleast_1d(observation["state.z"]).flatten()[-1]),
            ], dtype=np.float32)
            rot = np.array([
                float(np.atleast_1d(observation["state.roll"]).flatten()[-1]),
                float(np.atleast_1d(observation["state.pitch"]).flatten()[-1]),
                float(np.atleast_1d(observation["state.yaw"]).flatten()[-1]),
            ], dtype=np.float32)
            gripper = np.atleast_1d(observation["state.gripper"]).flatten().astype(np.float32)

            state_dict["state.eef_pos_absolute"] = torch.from_numpy(pos)
            state_dict["state.eef_rot_absolute"] = torch.from_numpy(rot)
            state_dict["state.gripper_close"] = torch.from_numpy(gripper)

        # Apply normalization if transform is available
        if self.state_action_transform is not None and state_dict:
            state_dict = self.state_action_transform(state_dict)

        # Concatenate and pad
        if state_dict:
            state_concat = torch.cat([
                state_dict.get("state.eef_pos_absolute", torch.zeros(3)),
                state_dict.get("state.eef_rot_absolute", torch.zeros(3)),
                state_dict.get("state.gripper_close", torch.zeros(2)),
            ])
        else:
            state_concat = torch.zeros(self.real_state_dim)

        # Pad to max_state_dim
        padded = torch.zeros(self.max_state_dim, dtype=torch.float32)
        padded[:len(state_concat)] = state_concat

        return padded.unsqueeze(0).to(self.device)  # (1, max_state_dim)

    def _get_prompts(self, observation: dict) -> list[str]:
        """Extract language prompts from observation."""
        for key in [
            "annotation.human.action.task_description",
            "language",
            "prompt",
            "task_description",
        ]:
            if key in observation:
                val = observation[key]
                if isinstance(val, list):
                    return val[:1]
                return [str(val)]
        return [""]

    def _postprocess_actions(self, actions: Tensor) -> dict[str, np.ndarray]:
        """
        Denormalize and extract real action dims from model output.

        Model outputs normalized actions in [-1, 1].
        transform.unapply() converts back to original scale.
        For LIBERO: first 7 dims = [dx, dy, dz, droll, dpitch, dyaw, gripper]

        Returns:
            Dict with grouped keys matching AlinVLA output format:
                "action.eef_pos_delta": (horizon, 3)
                "action.eef_rot_delta": (horizon, 3)
                "action.gripper_close": (horizon, 1)
        """
        actions_tensor = actions[0]  # (horizon, max_action_dim)
        horizon = actions_tensor.shape[0]

        # Split into individual action keys for denormalization
        action_dict = {
            "action.eef_pos_delta": actions_tensor[:, 0:3],
            "action.eef_rot_delta": actions_tensor[:, 3:6],
            "action.gripper_close": actions_tensor[:, 6:7],
        }

        # Denormalize using transform.unapply()
        if self.state_action_transform is not None:
            action_dict = self.state_action_transform.unapply(action_dict)

        # Convert to numpy
        return {
            "action.eef_pos_delta": action_dict["action.eef_pos_delta"].cpu().numpy(),
            "action.eef_rot_delta": action_dict["action.eef_rot_delta"].cpu().numpy(),
            "action.gripper_close": action_dict["action.gripper_close"].cpu().numpy(),
        }

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_path: str,
        stats_path: Optional[str] = None,
        device: str = "cuda",
        denoising_steps: int = 16,
    ) -> "MimicVideoPolicy":
        """
        Load a policy from a training checkpoint.

        Args:
            checkpoint_path: Path to checkpoint directory
            stats_path: Path to stats.json (if not in checkpoint, e.g., dataset path)
            device: Device to load model on
            denoising_steps: Number of denoising steps for sampling
        """
        from mimic_video.mimic_video import MimicVideo

        ckpt_path = Path(checkpoint_path)

        # Load metadata
        metadata = _load_metadata(ckpt_path)
        config = {
            "dim_action": metadata.get("max_action_dim", 32),
            "dim_joint_state": metadata.get("max_state_dim", 64),
            "denoising_steps": denoising_steps,
            "video_resolution": 224,
        }

        # Load weights from sharded safetensors
        state_dict = _load_safetensors_checkpoint(ckpt_path)

        # Auto-detect language injection from checkpoint weights
        inject_language_tokens = any(k.startswith("language_proj.") for k in state_dict)

        # Build model WITHOUT internal normalizer (using transform-based normalization)
        video_wrapper = _build_video_wrapper(ckpt_path)
        model = MimicVideo(
            dim=512,
            video_predict_wrapper=video_wrapper,
            action_chunk_len=16,
            dim_action=config["dim_action"],
            dim_joint_state=config["dim_joint_state"],
            num_video_viewpoints=2,
            model_output_clean=False,
            inject_language_tokens=inject_language_tokens,
            # No action_mean_std or joint_mean_std - using transform-based normalization
        )

        model.load_state_dict(state_dict, strict=False)

        # Load StateActionNormalize transform with stats
        state_action_transform = _build_state_action_transform(ckpt_path, stats_path)

        return cls(
            model=model,
            config=config,
            state_action_transform=state_action_transform,
            device=device,
        )

    def get_modality_config(self) -> dict:
        """Return modality configuration for compatibility with server interface."""
        return {
            "video_keys": ["video.front_view", "video.left_wrist_view"],
            "state_keys": ["state.eef_pos_absolute", "state.eef_rot_absolute", "state.gripper_close"],
            "action_keys": ["action.eef_pos_delta", "action.eef_rot_delta", "action.gripper_close"],
        }


def _load_metadata(ckpt_path: Path) -> dict:
    """Load metadata from experiment_cfg/metadata.json."""
    metadata_path = ckpt_path / "experiment_cfg" / "metadata.json"
    if not metadata_path.exists():
        metadata_path = ckpt_path.parent / "experiment_cfg" / "metadata.json"
    if metadata_path.exists():
        with open(metadata_path, "r") as f:
            return json.load(f)
    return {}


def _build_video_wrapper(ckpt_path: Path):
    """Build appropriate video wrapper based on checkpoint weights."""
    index_path = ckpt_path / "model.safetensors.index.json"
    if index_path.exists():
        with open(index_path, "r") as f:
            index = json.load(f)
        weight_map = index.get("weight_map", {})
        # Check if checkpoint has Cosmos2 weights
        has_cosmos2 = any("video_predict_wrapper" in k for k in weight_map)
    else:
        has_cosmos2 = True

    # Use Cosmos2PredictWrapper (the model we train with)
    from mimic_video.cosmos_predict import Cosmos2PredictWrapper
    return Cosmos2PredictWrapper(
        model_name="nvidia/Cosmos-Predict2-2B-Video2World",
        extract_layer=19,
    )


def _build_state_action_transform(
    ckpt_path: Path,
    stats_path: Optional[str] = None,
) -> Optional[StateActionNormalize]:
    """Build StateActionNormalize transform with stats from checkpoint or dataset."""

    # Define keys for LIBERO
    state_keys = ["state.eef_pos_absolute", "state.eef_rot_absolute", "state.gripper_close"]
    action_keys = ["action.eef_pos_delta", "action.eef_rot_delta", "action.gripper_close"]

    # Try to find stats.json
    stats = None

    # Option 1: stats_path provided directly
    if stats_path:
        stats_file = Path(stats_path)
        if stats_file.is_dir():
            stats_file = stats_file / "meta" / "stats.json"
        if stats_file.exists():
            with open(stats_file, "r") as f:
                stats = json.load(f)

    # Option 2: Look in checkpoint directory
    if stats is None:
        for candidate in [
            ckpt_path / "stats.json",
            ckpt_path / "experiment_cfg" / "stats.json",
            ckpt_path.parent / "stats.json",
        ]:
            if candidate.exists():
                with open(candidate, "r") as f:
                    stats = json.load(f)
                break

    if stats is None:
        print("Warning: No stats.json found. Normalization will be skipped.")
        return None

    # Load modality.json to get dimension slices
    modality_meta = None
    for candidate in [
        ckpt_path / "modality.json",
        ckpt_path / "experiment_cfg" / "modality.json",
        ckpt_path.parent / "modality.json",
    ]:
        if candidate.exists():
            with open(candidate, "r") as f:
                modality_meta = json.load(f)
            break

    # Build per-key stats
    per_key_stats = _build_per_key_stats(stats, modality_meta)

    # Create transform
    transform = StateActionNormalize(
        apply_to=state_keys + action_keys,
        normalization_modes={key: "min_max" for key in state_keys + action_keys},
    )
    transform.set_stats(per_key_stats)

    return transform


def _build_per_key_stats(stats: dict, modality_meta: Optional[dict]) -> dict:
    """Build per-key stats by slicing concatenated stats using modality metadata."""
    per_key_stats = {}

    # Default dimension mapping for LIBERO if no modality.json
    default_mapping = {
        "state": {
            "eef_pos_absolute": (0, 3),
            "eef_rot_absolute": (3, 6),
            "gripper_close": (6, 8),
        },
        "action": {
            "eef_pos_delta": (0, 3),
            "eef_rot_delta": (3, 6),
            "gripper_close": (6, 7),
        },
    }

    # Map from modality type to stats.json key
    stats_key_map = {
        "state": "observation.state",
        "action": "action",
    }

    for modality_type, stats_key in stats_key_map.items():
        if stats_key not in stats:
            continue

        concat_stats = stats[stats_key]

        # Get slice mapping from modality_meta or default
        if modality_meta and modality_type in modality_meta:
            slice_mapping = {
                subkey: (meta.get("start"), meta.get("end"))
                for subkey, meta in modality_meta[modality_type].items()
                if meta.get("start") is not None and meta.get("end") is not None
            }
        else:
            slice_mapping = default_mapping.get(modality_type, {})

        for subkey, (start, end) in slice_mapping.items():
            full_key = f"{modality_type}.{subkey}"
            per_key_stats[full_key] = {}

            for stat_name in ["min", "max", "mean", "std", "q01", "q99"]:
                if stat_name in concat_stats:
                    values = concat_stats[stat_name]
                    per_key_stats[full_key][stat_name] = values[start:end]

    return per_key_stats


def _load_safetensors_checkpoint(ckpt_path: Path) -> dict:
    """Load sharded safetensors checkpoint into a single state_dict."""
    from safetensors.torch import load_file

    index_path = ckpt_path / "model.safetensors.index.json"

    if index_path.exists():
        # Sharded format
        with open(index_path, "r") as f:
            index = json.load(f)
        weight_map = index["weight_map"]

        # Collect unique shard files
        shard_files = sorted(set(weight_map.values()))
        state_dict = {}
        for shard_name in shard_files:
            shard_path = ckpt_path / shard_name
            shard_dict = load_file(str(shard_path))
            state_dict.update(shard_dict)
        return state_dict

    # Single safetensors file
    single_path = ckpt_path / "model.safetensors"
    if single_path.exists():
        return load_file(str(single_path))

    # Fallback to model.pt
    model_pt = ckpt_path / "model.pt"
    if model_pt.exists():
        return torch.load(model_pt, map_location="cpu", weights_only=True)

    raise FileNotFoundError(f"No model weights found in {ckpt_path}")
