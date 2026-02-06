"""
Extensible data configuration system for mimic-video.

Following GR00T's BaseDataConfig pattern from gr00t/experiment/data_config.py.
Each dataset gets its own config subclass with appropriate keys, indices, and transforms.
"""

import importlib
import os
import sys
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from mimic_video.data.transforms import (
    Compose,
    StateActionNormalize,
    VideoColorJitter,
    VideoCrop,
    VideoResize,
    VideoToTensor,
)


@dataclass
class ModalityConfig:
    """Config for a single modality: delta_indices + modality_keys."""

    delta_indices: list[int]
    modality_keys: list[str]


@dataclass
class BaseDataConfig(ABC):
    """
    Abstract base for dataset-specific configurations.
    Subclass for each dataset (LIBERO, RoboCasa, etc.).
    Mirrors GR00T's BaseDataConfig.
    """

    video_keys: list[str] = field(default_factory=list)
    state_keys: list[str] = field(default_factory=list)
    action_keys: list[str] = field(default_factory=list)
    language_keys: list[str] = field(default_factory=list)
    observation_indices: list[int] = field(default_factory=lambda: [0])
    action_indices: list[int] = field(default_factory=lambda: list(range(16)))
    future_video_indices: list[int] = field(default_factory=list)  # e.g., [1] or [16] for Algorithm 2

    def modality_config(self) -> dict[str, ModalityConfig]:
        """Build ModalityConfig for each modality."""
        configs = {
            "video": ModalityConfig(
                delta_indices=self.observation_indices,
                modality_keys=self.video_keys,
            ),
            "state": ModalityConfig(
                delta_indices=self.observation_indices,
                modality_keys=self.state_keys,
            ),
            "action": ModalityConfig(
                delta_indices=self.action_indices,
                modality_keys=self.action_keys,
            ),
            "language": ModalityConfig(
                delta_indices=self.observation_indices,
                modality_keys=self.language_keys,
            ),
        }
        # Add future_video modality if future_video_indices is set
        if self.future_video_indices:
            configs["future_video"] = ModalityConfig(
                delta_indices=self.future_video_indices,
                modality_keys=self.video_keys,  # Use same keys as video
            )
        return configs

    @abstractmethod
    def transform(self, training: bool = True) -> Compose:
        """Build the transform pipeline for this dataset."""
        pass


class LiberoDataConfig(BaseDataConfig):
    """LIBERO dataset configuration."""

    def __init__(
        self,
        num_frames: int = 1,
        video_resolution: int = 224,
        num_future_frames: int = 0,
        future_frame_start: int = 1,
    ):
        # Set observation_indices based on num_frames (past frames)
        observation_indices = list(range(-num_frames + 1, 1))  # e.g. [0] for 1, [-1, 0] for 2

        # Set future_video_indices if num_future_frames > 0
        # Algorithm 2: future frames are at positive indices relative to current step
        # future_frame_start allows specifying which timestep to start from (e.g., 16 for distant future)
        if num_future_frames > 0:
            future_video_indices = list(range(future_frame_start, future_frame_start + num_future_frames))
        else:
            future_video_indices = []

        super().__init__(
            video_keys=["video.front_view", "video.left_wrist_view"],
            state_keys=[
                "state.eef_pos_absolute",
                "state.eef_rot_absolute",
                "state.gripper_close",
            ],
            action_keys=[
                "action.eef_pos_delta",
                "action.eef_rot_delta",
                "action.gripper_close",
            ],
            language_keys=["annotation.human.action.task_description"],
            observation_indices=observation_indices,
            action_indices=list(range(16)),
            future_video_indices=future_video_indices,
        )
        self.video_resolution = video_resolution
        self.num_future_frames = num_future_frames

    def transform(self, training: bool = True) -> Compose:
        transforms = [
            # Video: uint8 HWC → float32 CHW [0,1]
            VideoToTensor(apply_to=self.video_keys),
            # Crop: random 95% crop (training) or center crop (eval)
            VideoCrop(apply_to=self.video_keys, scale=0.95, random=training),
            # Resize to target resolution
            VideoResize(
                apply_to=self.video_keys,
                height=self.video_resolution,
                width=self.video_resolution,
            ),
        ]
        # Color jitter only during training
        if training:
            transforms.append(
                VideoColorJitter(
                    apply_to=self.video_keys,
                    brightness=0.3,
                    contrast=0.4,
                    saturation=0.5,
                    hue=0.08,
                )
            )
        # Note: State/action normalization is handled by MimicVideo's internal
        # Normalizer (mean/std from stats.json). Do NOT apply StateActionNormalize
        # here, as it would create a mismatch between training and inference:
        # - Training: StateActionNormalize + Model.Normalizer (double normalization)
        # - Inference: Model.Normalizer only (policy provides raw values)
        return Compose(transforms)


# Registry of built-in data configs
DATA_CONFIG_MAP = {
    "libero": LiberoDataConfig,
}


def import_external_data_config(data_config_str: str) -> Optional[BaseDataConfig]:
    """
    Import and instantiate an external data configuration class.
    Format: "module_path:ClassName" (e.g., "my_configs:RobotConfig")
    """
    if ":" not in data_config_str:
        return None

    current_dir = str(Path(os.getcwd()).absolute())
    if current_dir not in sys.path:
        sys.path.insert(0, current_dir)

    module_path, class_name = data_config_str.split(":", 1)
    if not module_path or not class_name:
        raise ValueError(f"Invalid format: '{data_config_str}'. Use 'module:ClassName'")

    module = importlib.import_module(module_path)
    if not hasattr(module, class_name):
        available = [
            n for n in dir(module) if not n.startswith("_") and isinstance(getattr(module, n), type)
        ]
        raise AttributeError(
            f"Class '{class_name}' not found in '{module_path}'. Available: {available}"
        )

    config_cls = getattr(module, class_name)
    return config_cls()


def load_data_config(name: str, **kwargs) -> BaseDataConfig:
    """
    Load data config by name ('libero') or external path ('module:ClassName').
    """
    # Try external import first
    external = import_external_data_config(name)
    if external is not None:
        return external

    if name not in DATA_CONFIG_MAP:
        raise ValueError(f"Unknown data config: '{name}'. Available: {list(DATA_CONFIG_MAP.keys())}")

    return DATA_CONFIG_MAP[name](**kwargs)
