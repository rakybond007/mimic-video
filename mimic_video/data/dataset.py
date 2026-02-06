"""
LeRobot-format dataset for mimic-video.

Two-level design: BaseLeRobotDataset handles format loading and padding,
subclasses (e.g., LeRobotLiberoDataset) provide dataset-specific defaults.

NOTE: pandas is imported lazily (inside methods) so that modules depending
on mimic_video.data can be imported without pandas installed.

Reference: gr00t/data/dataset.py (LeRobotSingleDataset)

Directory structure expected:
    dataset_path/
    ├── meta/
    │   ├── episodes.jsonl
    │   ├── stats.json
    │   ├── tasks.jsonl
    │   ├── info.json
    │   └── modality.json
    ├── data/
    │   └── chunk-*/*.parquet
    └── videos/
        └── chunk-*/*.mp4
"""

import json
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from mimic_video.data.data_config import BaseDataConfig, LiberoDataConfig, ModalityConfig
from mimic_video.data.normalization import load_stats
from mimic_video.data.transforms import Compose


@dataclass
class ModalityConfig:
    """Config for a modality: delta_indices + modality_keys."""

    delta_indices: list[int]
    modality_keys: list[str]


class BaseLeRobotDataset(Dataset):
    """
    Base dataset for any LeRobot-format data. Handles:
    - Parquet loading, video decoding, episode indexing
    - Delta-index based sampling (observation_indices, action_indices)
    - Padding (first/last value for boundary samples)
    - Padding to max_state_dim/max_action_dim with masks
    """

    def __init__(
        self,
        dataset_path: str,
        data_config: BaseDataConfig,
        max_state_dim: int = 64,
        max_action_dim: int = 32,
        training: bool = True,
    ):
        self.dataset_path = Path(dataset_path)
        if not self.dataset_path.exists():
            raise FileNotFoundError(f"Dataset path {dataset_path} does not exist")

        self.data_config = data_config
        self.max_state_dim = max_state_dim
        self.max_action_dim = max_action_dim
        self.training = training

        # Build modality configs and transforms
        self.modality_configs = data_config.modality_config()
        self.transforms = data_config.transform(training=training)

        # Parse delta indices per key
        # Note: video and future_video share the same video_keys, so we need to use
        # different key prefixes to avoid overwriting
        self._delta_indices: dict[str, np.ndarray] = {}
        self._modality_keys: dict[str, list[str]] = defaultdict(list)
        for modality, config in self.modality_configs.items():
            for key in config.modality_keys:
                # Use "future_{key}" for future_video to avoid overwriting video keys
                storage_key = f"future_{key}" if modality == "future_video" else key
                self._delta_indices[storage_key] = np.array(config.delta_indices)
                self._modality_keys[modality].append(key)

        # Load dataset metadata
        self._load_metadata()

        # Build flat index: list of (episode_index, base_step)
        self._all_steps = self._build_step_index()

        # Load stats for normalization
        self._stats = self._load_stats()

        # Set stats on transforms
        self._set_transform_stats()

        # Cache for current trajectory data
        self._curr_traj_id = None
        self._curr_traj_data = None

    def _load_metadata(self):
        """Load episode metadata, info, and tasks."""
        # Episodes
        episodes_path = self.dataset_path / "meta" / "episodes.jsonl"
        with open(episodes_path, "r") as f:
            episodes = [json.loads(line) for line in f]
        self._episode_ids = np.array([e["episode_index"] for e in episodes])
        self._episode_lengths = np.array([e["length"] for e in episodes])

        # Info
        info_path = self.dataset_path / "meta" / "info.json"
        with open(info_path, "r") as f:
            self._info = json.load(f)
        self._data_path_pattern = self._info.get("data_path", "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet")
        self._video_path_pattern = self._info.get("video_path", "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4")
        self._chunk_size = self._info.get("chunks_size", 1000)

        # Tasks
        tasks_path = self.dataset_path / "meta" / "tasks.jsonl"
        if tasks_path.exists():
            with open(tasks_path, "r") as f:
                tasks = [json.loads(line) for line in f]
            self._tasks = {t["task_index"]: t.get("task", t.get("task_description", "")) for t in tasks}
        else:
            self._tasks = {}

        # Modality metadata (optional)
        modality_path = self.dataset_path / "meta" / "modality.json"
        if modality_path.exists():
            with open(modality_path, "r") as f:
                self._modality_meta = json.load(f)
        else:
            self._modality_meta = {}

        # Build modality type → parquet column name mapping from info.json features
        # e.g. "state" → "observation.state", "action" → "action"
        self._feature_columns: dict[str, str] = {}
        features = self._info.get("features", {})
        for feat_name, feat_info in features.items():
            dtype = feat_info.get("dtype", "")
            if dtype in ("float32", "float64", "int32", "int64"):
                if "state" in feat_name.lower():
                    self._feature_columns["state"] = feat_name
                elif feat_name.lower().startswith("action"):
                    self._feature_columns["action"] = feat_name

    def _build_step_index(self) -> list[tuple[int, int]]:
        """Build flat (episode_id, base_step) index over all episodes."""
        all_steps = []
        max_delta = max(
            int(di.max()) for di in self._delta_indices.values() if len(di) > 0
        )
        for ep_id, ep_len in zip(self._episode_ids, self._episode_lengths):
            # We can sample any base_index from 0 to ep_len - 1
            # Padding handles out-of-range delta indices
            for step in range(ep_len):
                all_steps.append((int(ep_id), step))
        return all_steps

    def _load_stats(self) -> dict:
        """Load dataset statistics for normalization."""
        try:
            return load_stats(str(self.dataset_path))
        except FileNotFoundError:
            return {}

    def _set_transform_stats(self):
        """Pass stats to the normalization transform."""
        if self._stats and self.transforms:
            for t in self.transforms.transforms:
                if hasattr(t, "set_stats"):
                    t.set_stats(self._stats)

    @property
    def stats(self) -> dict:
        return self._stats

    def __len__(self) -> int:
        return len(self._all_steps)

    def __getitem__(self, index: int) -> dict:
        ep_id, base_step = self._all_steps[index]
        sample = self._get_step_data(ep_id, base_step)
        # Apply transforms (video crop/resize/jitter, state/action normalize)
        sample = self.transforms(sample, stats=self._stats)
        # Post-process: stack video views, pad state/action, create masks
        return self._postprocess(sample)

    def _get_trajectory_data(self, ep_id: int):
        """Load parquet data for an episode (cached)."""
        if self._curr_traj_id == ep_id and self._curr_traj_data is not None:
            return self._curr_traj_data

        chunk_index = ep_id // self._chunk_size
        parquet_path = self.dataset_path / self._data_path_pattern.format(
            episode_chunk=chunk_index, episode_index=ep_id
        )
        if not parquet_path.exists():
            raise FileNotFoundError(f"Parquet file not found: {parquet_path}")

        self._curr_traj_data = pd.read_parquet(parquet_path)
        self._curr_traj_id = ep_id
        return self._curr_traj_data

    def _get_step_data(self, ep_id: int, base_step: int) -> dict:
        """Load raw data for a single step across all modalities."""
        traj_data = self._get_trajectory_data(ep_id)
        traj_len = len(traj_data)
        data = {}

        for modality, keys in self._modality_keys.items():
            for key in keys:
                # Use "future_{key}" for future_video delta_indices lookup
                storage_key = f"future_{key}" if modality == "future_video" else key
                delta_indices = self._delta_indices[storage_key]
                step_indices = base_step + delta_indices

                if modality == "video":
                    data[key] = self._get_video(ep_id, key, step_indices, traj_len)
                elif modality == "future_video":
                    # Store future video with "future_" prefix
                    data[f"future_{key}"] = self._get_video(ep_id, key, step_indices, traj_len)
                elif modality == "language":
                    data[key] = self._get_language(traj_data, key, base_step)
                else:
                    # state or action
                    data[key] = self._get_state_or_action(traj_data, key, step_indices, traj_len)

        return data

    def _get_video(
        self, ep_id: int, key: str, step_indices: np.ndarray, traj_len: int
    ) -> np.ndarray:
        """Load video frames at specified step indices."""
        chunk_index = ep_id // self._chunk_size

        # Determine video key for path
        # Map our key (e.g. "video.front_view") to the actual video path
        video_key_for_path = key  # e.g. "video.front_view"

        # Check if modality.json provides an original_key mapping
        if self._modality_meta and "video" in self._modality_meta:
            video_meta = self._modality_meta["video"]
            # key without "video." prefix
            short_key = key.replace("video.", "")
            if short_key in video_meta:
                orig = video_meta[short_key].get("original_key")
                if orig:
                    video_key_for_path = orig

        video_path = self.dataset_path / self._video_path_pattern.format(
            episode_chunk=chunk_index,
            video_key=video_key_for_path,
            episode_index=ep_id,
        )

        if not video_path.exists():
            # Fallback: try with the key as-is in the path
            video_path = self.dataset_path / "videos" / f"chunk-{chunk_index:03d}" / key / f"episode_{ep_id:06d}.mp4"

        # Clamp indices to valid range for actual loading
        valid_indices = np.clip(step_indices, 0, traj_len - 1)

        try:
            frames = self._decode_video_frames(str(video_path), valid_indices)
        except Exception:
            # If video decoding fails, return zeros
            frames = np.zeros((len(step_indices), 224, 224, 3), dtype=np.uint8)

        # Apply boundary padding: use first/last frame for out-of-range
        for i, si in enumerate(step_indices):
            if si < 0:
                frames[i] = frames[0] if len(frames) > 0 else np.zeros_like(frames[0])
            elif si >= traj_len:
                frames[i] = frames[-1] if len(frames) > 0 else np.zeros_like(frames[0])

        return frames  # (T, H, W, C)

    def _decode_video_frames(self, video_path: str, frame_indices: np.ndarray) -> np.ndarray:
        """Decode specific frames from a video file."""
        try:
            import torchcodec

            decoder = torchcodec.decoders.VideoDecoder(video_path)
            frames = []
            for idx in frame_indices:
                frame = decoder[int(idx)]
                # torchcodec returns (C, H, W) tensor
                frame_np = frame.permute(1, 2, 0).numpy().astype(np.uint8)
                frames.append(frame_np)
            return np.stack(frames)
        except (ImportError, Exception):
            pass

        try:
            import cv2

            cap = cv2.VideoCapture(video_path)
            frames = []
            unique_indices = sorted(set(frame_indices.tolist()))
            frame_map = {}
            current_frame = 0
            for idx in unique_indices:
                cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
                ret, frame = cap.read()
                if ret:
                    frame_map[idx] = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                else:
                    frame_map[idx] = np.zeros((224, 224, 3), dtype=np.uint8)
            cap.release()

            return np.stack([frame_map[int(i)] for i in frame_indices])
        except ImportError:
            pass

        try:
            import imageio

            reader = imageio.get_reader(video_path)
            all_frames = [f for f in reader]
            reader.close()
            frames = []
            for idx in frame_indices:
                idx = min(int(idx), len(all_frames) - 1)
                frames.append(all_frames[idx])
            return np.stack(frames)
        except ImportError:
            raise RuntimeError(
                "No video decoder available. Install torchcodec, opencv-python, or imageio."
            )

    def _get_state_or_action(
        self, traj_data, key: str, step_indices: np.ndarray, traj_len: int
    ) -> np.ndarray:
        """Load state or action values at specified step indices with padding."""
        # Map our key to the column in parquet
        # The modality.json may map e.g. "state.eef_pos_absolute" -> "observation.state"
        col_name, col_slice = self._resolve_column(key)

        if col_name not in traj_data.columns:
            # Try without prefix transformation
            simple_key = key.split(".")[-1]
            for col in traj_data.columns:
                if simple_key in col:
                    col_name = col
                    break
            else:
                raise KeyError(f"Column for key '{key}' not found in parquet. Available: {list(traj_data.columns)}")

        # Load the raw array
        raw_values = traj_data[col_name].values
        # Stack into array: each element may be a scalar or list
        try:
            raw_array = np.vstack([np.atleast_1d(np.asarray(v, dtype=np.float32)) for v in raw_values])
        except ValueError:
            raw_array = np.array([np.asarray(v, dtype=np.float32).flatten() for v in raw_values])

        # Apply column slice if specified
        if col_slice is not None:
            raw_array = raw_array[:, col_slice[0]:col_slice[1]]

        # Retrieve with padding
        result = self._retrieve_and_pad(raw_array, step_indices, traj_len)
        return result  # (T, dim)

    def _resolve_column(self, key: str) -> tuple[str, tuple[int, int] | None]:
        """Resolve a modality key to a parquet column name and optional slice.

        Uses modality.json metadata if available.
        """
        if not self._modality_meta:
            return key, None

        # Determine modality type and subkey
        parts = key.split(".", 1)
        if len(parts) != 2:
            return key, None

        modality_type, subkey = parts

        meta_section = self._modality_meta.get(modality_type, {})
        if subkey in meta_section:
            meta = meta_section[subkey]
            original_key = meta.get("original_key")
            if original_key is None:
                # Infer column name from info.json features mapping
                original_key = self._feature_columns.get(modality_type, key)
            start = meta.get("start")
            end = meta.get("end")
            col_slice = (start, end) if start is not None and end is not None else None
            return original_key, col_slice

        return key, None

    def _get_language(self, traj_data, key: str, base_step: int) -> str:
        """Get language annotation for a step."""
        # Try to find the task description in traj_data or via tasks mapping
        col_name, _ = self._resolve_column(key)

        # Check if column exists in parquet
        if col_name in traj_data.columns:
            value = traj_data[col_name].iloc[base_step]
            if isinstance(value, str):
                return value

        # Try via task_index column
        if "task_index" in traj_data.columns:
            task_idx = int(traj_data["task_index"].iloc[base_step])
            if task_idx in self._tasks:
                return self._tasks[task_idx]

        # Fallback: check if annotation column exists
        for col in traj_data.columns:
            if "task" in col.lower() or "annotation" in col.lower() or "language" in col.lower():
                val = traj_data[col].iloc[base_step]
                if isinstance(val, str):
                    return val

        return ""

    def _retrieve_and_pad(
        self, array: np.ndarray, step_indices: np.ndarray, max_length: int
    ) -> np.ndarray:
        """Retrieve data at step indices with first/last value padding for out-of-range."""
        front_pad = step_indices < 0
        end_pad = step_indices >= max_length
        valid = ~(front_pad | end_pad)

        # Clamp to valid range for indexing
        clamped = np.clip(step_indices, 0, max_length - 1)
        result = array[clamped]

        # Apply padding
        if front_pad.any():
            result[front_pad] = array[0]
        if end_pad.any():
            result[end_pad] = array[-1]

        return result

    def _postprocess(self, sample: dict) -> dict:
        """
        Stack video views, pad state/action to max dims, create masks.

        Returns dict with:
            video: (V, T, C, H, W) float32 tensor
            future_video: (V, T_future, C, H, W) float32 tensor (if future_video_indices set)
            actions: (horizon, max_action_dim) float32 tensor
            action_mask: (max_action_dim,) bool tensor
            joint_state: (T, max_state_dim) float32 tensor
            state_mask: (max_state_dim,) bool tensor
            prompt: str
        """
        # Stack video views: (V, T, C, H, W)
        video_tensors = []
        for vk in self.data_config.video_keys:
            v = sample[vk]
            if isinstance(v, np.ndarray):
                v = torch.from_numpy(v).float()
            if v.ndim == 3:
                v = v.unsqueeze(0)  # (C, H, W) → (1, C, H, W)
            video_tensors.append(v)
        video = torch.stack(video_tensors, dim=0)  # (V, T, C, H, W)

        # Stack future video views if available (Algorithm 2)
        # Note: future_video keys are not transformed, so we apply same processing as video
        future_video = None
        if self.data_config.future_video_indices:
            future_video_tensors = []
            for vk in self.data_config.video_keys:
                fv_key = f"future_{vk}"
                if fv_key in sample:
                    fv = sample[fv_key]
                    # fv shape: (T, H, W, C) raw from video loader
                    if isinstance(fv, np.ndarray):
                        # Apply same transforms as video: HWC -> CHW, normalize, resize
                        fv = torch.from_numpy(fv).float()
                        if fv.ndim == 3:
                            fv = fv.unsqueeze(0)  # (H, W, C) → (1, H, W, C)
                        # (T, H, W, C) -> (T, C, H, W)
                        if fv.shape[-1] == 3:
                            fv = fv.permute(0, 3, 1, 2)
                        # Normalize to [0, 1]
                        if fv.max() > 1.0:
                            fv = fv / 255.0
                        # Resize to video_resolution
                        if hasattr(self.data_config, 'video_resolution'):
                            res = self.data_config.video_resolution
                            fv = torch.nn.functional.interpolate(
                                fv, size=(res, res), mode='bilinear', align_corners=False
                            )
                    future_video_tensors.append(fv)
            if future_video_tensors:
                future_video = torch.stack(future_video_tensors, dim=0)  # (V, T_future, C, H, W)

        # Concatenate and pad state
        state_parts = []
        for sk in self.data_config.state_keys:
            s = sample[sk]
            if isinstance(s, np.ndarray):
                s = torch.from_numpy(s).float()
            if s.ndim == 1:
                s = s.unsqueeze(0)  # (dim,) → (1, dim)
            state_parts.append(s)
        state_concat = torch.cat(state_parts, dim=-1)  # (T, real_state_dim)
        real_state_dim = state_concat.shape[-1]
        T_state = state_concat.shape[0]

        # Pad to max_state_dim
        joint_state = torch.zeros(T_state, self.max_state_dim)
        joint_state[:, :real_state_dim] = state_concat
        state_mask = torch.zeros(self.max_state_dim, dtype=torch.bool)
        state_mask[:real_state_dim] = True

        # Concatenate and pad action
        action_parts = []
        for ak in self.data_config.action_keys:
            a = sample[ak]
            if isinstance(a, np.ndarray):
                a = torch.from_numpy(a).float()
            if a.ndim == 1:
                a = a.unsqueeze(0)  # (dim,) → (1, dim)
            action_parts.append(a)
        action_concat = torch.cat(action_parts, dim=-1)  # (horizon, real_action_dim)
        real_action_dim = action_concat.shape[-1]
        horizon = action_concat.shape[0]

        # Pad to max_action_dim
        actions = torch.zeros(horizon, self.max_action_dim)
        actions[:, :real_action_dim] = action_concat
        action_mask = torch.zeros(self.max_action_dim, dtype=torch.bool)
        action_mask[:real_action_dim] = True

        # Language prompt
        prompt = ""
        for lk in self.data_config.language_keys:
            if lk in sample and isinstance(sample[lk], str):
                prompt = sample[lk]
                break

        result = {
            "video": video,
            "actions": actions,
            "action_mask": action_mask,
            "joint_state": joint_state,
            "state_mask": state_mask,
            "prompt": prompt,
        }

        # Add future_video if available (Algorithm 2)
        if future_video is not None:
            result["future_video"] = future_video

        return result


class LeRobotLiberoDataset(BaseLeRobotDataset):
    """LIBERO-specific dataset. Uses LiberoDataConfig by default."""

    def __init__(
        self,
        dataset_path: str,
        data_config: BaseDataConfig | None = None,
        num_frames: int = 1,
        video_resolution: int = 224,
        **kwargs,
    ):
        if data_config is None:
            data_config = LiberoDataConfig(
                num_frames=num_frames,
                video_resolution=video_resolution,
            )
        super().__init__(dataset_path, data_config, **kwargs)
