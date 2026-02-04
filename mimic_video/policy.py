"""
MimicVideoPolicy: wraps MimicVideo for inference.

Handles: raw observations -> preprocess -> model.sample() -> extract actions.

The MimicVideo model handles normalization internally via its built-in
Normalizer objects (action_normalizer, joint_normalizer). These are registered
as buffers and saved/loaded with checkpoints. So the policy does NOT need
to do separate normalize/denormalize — just provide raw inputs and extract
raw outputs.

Reference: gr00t/model/policy.py (GR00TPolicy)
"""

import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor


class MimicVideoPolicy:
    """
    Wraps MimicVideo for inference.

    Preprocessing:
      - video: uint8 HWC -> float32 CHW [0,1], resize to video_resolution
      - state: raw values concatenated, padded to max_state_dim
    Post-processing:
      - model.sample() returns denormalized actions (inverse_normalize is done internally)
      - extract first real_action_dim dims from the 32-dim output
    """

    # LIBERO produces 8-dim state: eef_pos(3) + axis_angle(3) + gripper_qpos(2)
    LIBERO_REAL_STATE_DIM = 8
    LIBERO_REAL_ACTION_DIM = 7

    def __init__(
        self,
        model,
        config: dict,
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

    def get_action(self, observation: dict[str, Any]) -> dict[str, np.ndarray]:
        """
        Get actions from observations.

        Args:
            observation: Dict with:
                "video.front_view": (T, H, W, 3) uint8 or "video.image"
                "video.left_wrist_view": (T, H, W, 3) uint8 or "video.wrist_image"
                "state.eef_pos_absolute": (3,) float or individual state.x/y/z
                "state.eef_rot_absolute": (3,) float or individual state.roll/pitch/yaw
                "state.gripper_close": (N,) float or state.gripper
                "annotation.human.action.task_description": [str]

        Returns:
            Dict with:
                "action.eef_pos_delta": (horizon, 3)
                "action.eef_rot_delta": (horizon, 3)
                "action.gripper_close": (horizon, 1)
        """
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            # 1. Preprocess video
            video = self._preprocess_video(observation)  # (1, V, T, C, H, W)

            # 2. Preprocess state
            joint_state = self._preprocess_state(observation)  # (1, max_state_dim)

            # 3. Get prompt
            prompts = self._get_prompts(observation)

            # 4. Sample actions — model handles normalization internally
            actions = self.model.sample(
                steps=self.denoising_steps,
                batch_size=1,
                video=video,
                joint_state=joint_state,
                prompts=prompts,
                disable_progress_bar=True,
            )  # (1, action_chunk_len, max_action_dim)

            # 5. Extract real action dims
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
        Preprocess state observations and pad to max_state_dim.

        The model normalizes internally via joint_normalizer, so we provide raw values.
        Construct 8-dim state: eef_pos(3) + axis_angle(3) + gripper_qpos(2).
        """
        state_parts = []

        # Try grouped keys first (server/client mode sends these)
        if "state.eef_pos_absolute" in observation:
            state_parts.append(np.atleast_1d(observation["state.eef_pos_absolute"]).flatten())
            state_parts.append(np.atleast_1d(observation["state.eef_rot_absolute"]).flatten())
            state_parts.append(np.atleast_1d(observation["state.gripper_close"]).flatten())
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
            gripper = np.atleast_1d(observation["state.gripper"]).flatten()
            state_parts.extend([pos, rot, gripper])

        if not state_parts:
            state_concat = np.zeros(self.real_state_dim, dtype=np.float32)
        else:
            state_concat = np.concatenate(state_parts).astype(np.float32)

        # Pad to max_state_dim
        padded = np.zeros(self.max_state_dim, dtype=np.float32)
        padded[:len(state_concat)] = state_concat

        return torch.from_numpy(padded).unsqueeze(0).to(self.device)  # (1, max_state_dim)

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
        Extract real action dims from model output.

        model.sample() already inverse-normalizes, so output is in original space.
        For LIBERO: first 7 dims = [dx, dy, dz, droll, dpitch, dyaw, gripper]

        Returns:
            Dict with grouped keys matching AlinVLA output format:
                "action.eef_pos_delta": (horizon, 3)
                "action.eef_rot_delta": (horizon, 3)
                "action.gripper_close": (horizon, 1)
        """
        actions_np = actions[0].cpu().numpy()  # (horizon, max_action_dim)
        real = actions_np[:, :self.real_action_dim]  # (horizon, 7)

        return {
            "action.eef_pos_delta": real[:, 0:3],      # (horizon, 3)
            "action.eef_rot_delta": real[:, 3:6],       # (horizon, 3)
            "action.gripper_close": real[:, 6:7],       # (horizon, 1)
        }

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_path: str,
        device: str = "cuda",
        denoising_steps: int = 16,
    ) -> "MimicVideoPolicy":
        """
        Load a policy from a training checkpoint.

        Supports sharded safetensors format (model-XXXXX-of-XXXXX.safetensors
        with model.safetensors.index.json).

        Args:
            checkpoint_path: Path to checkpoint directory
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

        # Extract normalizer stats from checkpoint so the model creates
        # proper Normalizer objects (register_buffer) before load_state_dict.
        # Without this, strict=False silently drops the 4 normalizer buffer
        # keys and all normalize/inverse_normalize calls become no-ops.
        action_mean_std = None
        joint_mean_std = None
        if "action_normalizer.mean" in state_dict and "action_normalizer.std" in state_dict:
            action_mean_std = torch.stack([
                state_dict["action_normalizer.mean"],
                state_dict["action_normalizer.std"],
            ])
        if "joint_normalizer.mean" in state_dict and "joint_normalizer.std" in state_dict:
            joint_mean_std = torch.stack([
                state_dict["joint_normalizer.mean"],
                state_dict["joint_normalizer.std"],
            ])

        # Build model (architecture must match training)
        video_wrapper = _build_video_wrapper(ckpt_path)
        model = MimicVideo(
            dim=512,
            video_predict_wrapper=video_wrapper,
            action_chunk_len=16,
            dim_action=config["dim_action"],
            dim_joint_state=config["dim_joint_state"],
            num_video_viewpoints=2,
            model_output_clean=False,
            action_mean_std=action_mean_std,
            joint_mean_std=joint_mean_std,
        )

        model.load_state_dict(state_dict, strict=False)

        return cls(model=model, config=config, device=device)

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
