"""
Modular transforms for video, state, and action data.

Follows GR00T's transform chain pattern but simplified for MimicVideo
(no backbone-specific token creation).
"""

from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor


class Compose:
    """Chain of transforms applied to a sample dict."""

    def __init__(self, transforms: list):
        self.transforms = transforms

    def __call__(self, sample: dict, stats: Optional[dict] = None) -> dict:
        for t in self.transforms:
            if isinstance(t, StateActionNormalize):
                sample = t(sample, stats=stats)
            else:
                sample = t(sample)
        return sample

    def __repr__(self):
        lines = [f"  {t}" for t in self.transforms]
        return "Compose([\n" + "\n".join(lines) + "\n])"


class VideoToTensor:
    """Convert uint8 HWC numpy video frames to float32 CHW tensors in [0, 1]."""

    def __init__(self, apply_to: list[str]):
        self.apply_to = apply_to

    def __call__(self, sample: dict) -> dict:
        for key in self.apply_to:
            if key not in sample:
                continue
            video = sample[key]
            if isinstance(video, np.ndarray):
                # video shape: (T, H, W, C) or (H, W, C)
                video = torch.from_numpy(video.copy()).float() / 255.0
                if video.ndim == 3:
                    # (H, W, C) → (C, H, W)
                    video = video.permute(2, 0, 1)
                elif video.ndim == 4:
                    # (T, H, W, C) → (T, C, H, W)
                    video = video.permute(0, 3, 1, 2)
            sample[key] = video
        return sample

    def __repr__(self):
        return f"VideoToTensor(apply_to={self.apply_to})"


class VideoCrop:
    """
    Random crop (training) or center crop (eval) at a given scale.
    Input: (T, C, H, W) or (C, H, W) float tensor.
    """

    def __init__(self, apply_to: list[str], scale: float = 0.95, random: bool = True):
        self.apply_to = apply_to
        self.scale = scale
        self.random = random

    def __call__(self, sample: dict) -> dict:
        for key in self.apply_to:
            if key not in sample:
                continue
            video = sample[key]
            has_time = video.ndim == 4
            if not has_time:
                video = video.unsqueeze(0)

            T, C, H, W = video.shape
            crop_h = int(H * self.scale)
            crop_w = int(W * self.scale)

            if self.random:
                top = torch.randint(0, H - crop_h + 1, (1,)).item()
                left = torch.randint(0, W - crop_w + 1, (1,)).item()
            else:
                top = (H - crop_h) // 2
                left = (W - crop_w) // 2

            video = video[:, :, top : top + crop_h, left : left + crop_w]
            if not has_time:
                video = video.squeeze(0)
            sample[key] = video
        return sample

    def __repr__(self):
        return f"VideoCrop(scale={self.scale}, random={self.random})"


class VideoResize:
    """Resize video frames to target (height, width)."""

    def __init__(
        self,
        apply_to: list[str],
        height: int = 224,
        width: int = 224,
        interpolation: str = "bilinear",
    ):
        self.apply_to = apply_to
        self.height = height
        self.width = width
        self.interpolation = interpolation

    def __call__(self, sample: dict) -> dict:
        for key in self.apply_to:
            if key not in sample:
                continue
            video = sample[key]
            has_time = video.ndim == 4
            if not has_time:
                video = video.unsqueeze(0)

            video = F.interpolate(
                video,
                size=(self.height, self.width),
                mode=self.interpolation,
                align_corners=False if self.interpolation != "nearest" else None,
            )

            if not has_time:
                video = video.squeeze(0)
            sample[key] = video
        return sample

    def __repr__(self):
        return f"VideoResize(h={self.height}, w={self.width})"


class VideoColorJitter:
    """Apply color jitter augmentation to video frames."""

    def __init__(
        self,
        apply_to: list[str],
        brightness: float = 0.3,
        contrast: float = 0.4,
        saturation: float = 0.5,
        hue: float = 0.08,
    ):
        self.apply_to = apply_to
        self.brightness = brightness
        self.contrast = contrast
        self.saturation = saturation
        self.hue = hue

    def _jitter(self, img: Tensor) -> Tensor:
        """Apply jitter to a single image (C, H, W)."""
        # Brightness
        if self.brightness > 0:
            factor = 1.0 + (torch.rand(1).item() * 2 - 1) * self.brightness
            img = img * factor

        # Contrast
        if self.contrast > 0:
            factor = 1.0 + (torch.rand(1).item() * 2 - 1) * self.contrast
            mean = img.mean(dim=(-2, -1), keepdim=True)
            img = (img - mean) * factor + mean

        # Saturation
        if self.saturation > 0 and img.shape[0] == 3:
            factor = 1.0 + (torch.rand(1).item() * 2 - 1) * self.saturation
            gray = img.mean(dim=0, keepdim=True)
            img = (img - gray) * factor + gray

        # Hue (simplified: rotate RGB channels slightly)
        if self.hue > 0 and img.shape[0] == 3:
            angle = (torch.rand(1).item() * 2 - 1) * self.hue * 3.14159
            cos_a, sin_a = torch.cos(torch.tensor(angle)), torch.sin(torch.tensor(angle))
            # Apply rotation in RG plane
            r, g, b = img[0], img[1], img[2]
            new_r = r * cos_a - g * sin_a
            new_g = r * sin_a + g * cos_a
            img = torch.stack([new_r, new_g, b], dim=0)

        return img.clamp(0.0, 1.0)

    def __call__(self, sample: dict) -> dict:
        for key in self.apply_to:
            if key not in sample:
                continue
            video = sample[key]
            has_time = video.ndim == 4
            if not has_time:
                video = video.unsqueeze(0)

            video = torch.stack([self._jitter(frame) for frame in video])

            if not has_time:
                video = video.squeeze(0)
            sample[key] = video
        return sample

    def __repr__(self):
        return (
            f"VideoColorJitter(brightness={self.brightness}, contrast={self.contrast}, "
            f"saturation={self.saturation}, hue={self.hue})"
        )


class StateActionNormalize:
    """
    Normalize state/action values using dataset statistics.

    Following AlinVLA/GR00T pattern:
    - min_max: maps to [-1, 1] range (not [0, 1])
    - Provides unapply() for inverse transform at inference
    - Stats are stored and reused for both apply and unapply
    """

    def __init__(
        self,
        apply_to: list[str],
        normalization_modes: Optional[dict[str, str]] = None,
    ):
        self.apply_to = apply_to
        self.normalization_modes = normalization_modes or {}
        self._stats = None
        self._tensors = {}  # Cached tensor stats for efficiency

    def set_stats(self, stats: dict):
        """Set statistics and cache as tensors."""
        self._stats = stats
        self._tensors = {}

        for key in self.apply_to:
            key_stats = stats.get(key)
            if key_stats is None:
                continue
            self._tensors[key] = {
                k: torch.tensor(v, dtype=torch.float32)
                for k, v in key_stats.items()
            }

    def __call__(self, sample: dict, stats: Optional[dict] = None) -> dict:
        """Apply normalization (forward transform)."""
        if stats is not None and stats != self._stats:
            self.set_stats(stats)

        if self._stats is None:
            return sample

        for key in self.apply_to:
            if key not in sample:
                continue
            if key not in self._tensors:
                continue

            mode = self.normalization_modes.get(key, "min_max")
            key_tensors = self._tensors[key]

            value = sample[key]
            if isinstance(value, np.ndarray):
                value = torch.from_numpy(value).float()

            if mode == "min_max":
                # AlinVLA style: map to [-1, 1]
                # Formula: 2 * (x - min) / (max - min) - 1
                vmin = key_tensors["min"]
                vmax = key_tensors["max"]
                denom = (vmax - vmin).clamp_min(1e-8)
                value = 2.0 * (value - vmin) / denom - 1.0
                # Clip to [-1, 1] for safety
                value = value.clamp(-1.0, 1.0)
            elif mode == "mean_std":
                mean = key_tensors["mean"]
                std = key_tensors["std"].clamp_min(1e-8)
                value = (value - mean) / std

            sample[key] = value
        return sample

    def unapply(self, sample: dict) -> dict:
        """Apply inverse normalization (for action output at inference)."""
        if self._stats is None:
            return sample

        for key in self.apply_to:
            if key not in sample:
                continue
            if key not in self._tensors:
                continue

            mode = self.normalization_modes.get(key, "min_max")
            key_tensors = self._tensors[key]

            value = sample[key]
            if isinstance(value, np.ndarray):
                value = torch.from_numpy(value).float()

            # Move stats to same device as value
            device = value.device if isinstance(value, torch.Tensor) else 'cpu'

            if mode == "min_max":
                # Inverse: (x + 1) / 2 * (max - min) + min
                vmin = key_tensors["min"].to(device)
                vmax = key_tensors["max"].to(device)
                value = (value + 1.0) / 2.0 * (vmax - vmin) + vmin
            elif mode == "mean_std":
                mean = key_tensors["mean"].to(device)
                std = key_tensors["std"].to(device)
                value = value * std + mean

            sample[key] = value
        return sample

    def __repr__(self):
        return f"StateActionNormalize(modes={self.normalization_modes})"
