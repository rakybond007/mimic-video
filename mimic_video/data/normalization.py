"""
Dataset statistics computation and normalization utilities.
"""

import json
from pathlib import Path
from typing import Optional, Union

import numpy as np
import torch
from torch import Tensor


def compute_dataset_stats(dataset_path: str, keys: Optional[list[str]] = None) -> dict:
    """
    Scan all parquet files in a LeRobot dataset and compute per-key statistics.

    Args:
        dataset_path: Path to the dataset root (containing data/ and meta/ dirs).
        keys: If provided, only compute stats for these keys.

    Returns:
        Dict mapping key names to {"min", "max", "mean", "std", "count"}.
    """
    import pyarrow.parquet as pq

    dataset_path = Path(dataset_path)
    data_dir = dataset_path / "data"

    # Collect all parquet files
    parquet_files = sorted(data_dir.rglob("*.parquet"))
    if not parquet_files:
        raise FileNotFoundError(f"No parquet files found in {data_dir}")

    # Initialize accumulators
    accumulators: dict[str, dict] = {}

    for pf in parquet_files:
        table = pq.read_table(pf)
        df = table.to_pandas()

        for col in df.columns:
            if keys is not None and col not in keys:
                continue
            # Only process numeric columns
            values = df[col].values
            if not np.issubdtype(values.dtype, np.number):
                # Try to handle list-like columns
                try:
                    values = np.stack(values)
                except (ValueError, TypeError):
                    continue

            if values.ndim == 1:
                values = values.reshape(-1, 1)

            if col not in accumulators:
                accumulators[col] = {
                    "sum": np.zeros(values.shape[1], dtype=np.float64),
                    "sum_sq": np.zeros(values.shape[1], dtype=np.float64),
                    "min": np.full(values.shape[1], np.inf),
                    "max": np.full(values.shape[1], -np.inf),
                    "count": 0,
                }

            acc = accumulators[col]
            acc["sum"] += values.sum(axis=0).astype(np.float64)
            acc["sum_sq"] += (values**2).sum(axis=0).astype(np.float64)
            acc["min"] = np.minimum(acc["min"], values.min(axis=0))
            acc["max"] = np.maximum(acc["max"], values.max(axis=0))
            acc["count"] += values.shape[0]

    # Compute final stats
    stats = {}
    for col, acc in accumulators.items():
        count = acc["count"]
        mean = acc["sum"] / count
        var = acc["sum_sq"] / count - mean**2
        std = np.sqrt(np.maximum(var, 0.0))

        stats[col] = {
            "min": acc["min"].tolist(),
            "max": acc["max"].tolist(),
            "mean": mean.tolist(),
            "std": std.tolist(),
            "count": int(count),
        }

    return stats


def load_stats(dataset_path: str) -> dict:
    """Load statistics from meta/stats.json in a dataset directory."""
    stats_path = Path(dataset_path) / "meta" / "stats.json"
    if not stats_path.exists():
        raise FileNotFoundError(f"Stats file not found: {stats_path}")

    with open(stats_path, "r") as f:
        return json.load(f)


def save_stats(stats: dict, output_path: str):
    """Save statistics to a JSON file."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(stats, f, indent=2)


def compute_normalizer_stats(
    dataset,
    num_samples: int = 5000,
    seed: int = 42,
) -> tuple[Tensor, Tensor]:
    """
    Compute action and joint_state mean/std from a dataset for MimicVideo's Normalizer.

    Returns:
        action_mean_std: (2, max_action_dim) tensor — [mean, std], padded dims get (0, 1)
        joint_mean_std:  (2, max_state_dim) tensor — [mean, std], padded dims get (0, 1)
    """
    rng = np.random.RandomState(seed)
    n = min(num_samples, len(dataset))
    indices = rng.choice(len(dataset), size=n, replace=False)

    all_actions = []
    all_states = []
    action_mask = None
    state_mask = None

    for idx in indices:
        sample = dataset[int(idx)]
        all_actions.append(sample["actions"])          # (horizon, max_action_dim)
        all_states.append(sample["joint_state"])       # (T, max_state_dim)
        if action_mask is None:
            action_mask = sample["action_mask"]        # (max_action_dim,) bool
            state_mask = sample.get("state_mask")      # (max_state_dim,) bool

    # Stack: (N, horizon, D) and (N, T, D)
    actions = torch.stack(all_actions)
    states = torch.stack(all_states)

    # Flatten time dims: (N*horizon, D) and (N*T, D)
    actions_flat = actions.reshape(-1, actions.shape[-1]).float()
    states_flat = states.reshape(-1, states.shape[-1]).float()

    # Compute mean/std per dimension
    action_mean = actions_flat.mean(dim=0)
    action_std = actions_flat.std(dim=0)
    state_mean = states_flat.mean(dim=0)
    state_std = states_flat.std(dim=0)

    # For padded (masked-out) dims, set mean=0 and std=1 so zeros stay zeros
    if action_mask is not None:
        pad = ~action_mask
        action_mean[pad] = 0.0
        action_std[pad] = 1.0
    if state_mask is not None:
        pad = ~state_mask
        state_mean[pad] = 0.0
        state_std[pad] = 1.0

    # Clamp std to avoid div-by-zero (some real dims may be constant)
    action_std = action_std.clamp(min=1e-6)
    state_std = state_std.clamp(min=1e-6)

    action_mean_std = torch.stack([action_mean, action_std])  # (2, max_action_dim)
    joint_mean_std = torch.stack([state_mean, state_std])     # (2, max_state_dim)

    return action_mean_std, joint_mean_std


def normalize(
    value: Union[np.ndarray, Tensor],
    key_stats: dict,
    mode: str = "min_max",
) -> Union[np.ndarray, Tensor]:
    """
    Normalize a value using precomputed statistics.

    Args:
        value: Input value to normalize.
        key_stats: Dict with "min", "max", "mean", "std" arrays.
        mode: "min_max" (to [0,1]) or "mean_std" (to N(0,1)).

    Returns:
        Normalized value (same type as input).
    """
    is_tensor = isinstance(value, Tensor)

    if mode == "min_max":
        vmin = np.array(key_stats["min"], dtype=np.float32)
        vmax = np.array(key_stats["max"], dtype=np.float32)
        denom = np.maximum(vmax - vmin, 1e-8)
        if is_tensor:
            vmin = torch.from_numpy(vmin).to(value.device)
            denom = torch.from_numpy(denom).to(value.device)
        return (value - vmin) / denom

    elif mode == "mean_std":
        mean = np.array(key_stats["mean"], dtype=np.float32)
        std = np.maximum(np.array(key_stats["std"], dtype=np.float32), 1e-8)
        if is_tensor:
            mean = torch.from_numpy(mean).to(value.device)
            std = torch.from_numpy(std).to(value.device)
        return (value - mean) / std

    else:
        raise ValueError(f"Unknown normalization mode: {mode}")


def denormalize(
    value: Union[np.ndarray, Tensor],
    key_stats: dict,
    mode: str = "min_max",
) -> Union[np.ndarray, Tensor]:
    """
    Inverse of normalize: recover original scale from normalized values.

    Args:
        value: Normalized value.
        key_stats: Dict with "min", "max", "mean", "std" arrays.
        mode: "min_max" or "mean_std".

    Returns:
        Denormalized value (same type as input).
    """
    is_tensor = isinstance(value, Tensor)

    if mode == "min_max":
        vmin = np.array(key_stats["min"], dtype=np.float32)
        vmax = np.array(key_stats["max"], dtype=np.float32)
        denom = np.maximum(vmax - vmin, 1e-8)
        if is_tensor:
            vmin = torch.from_numpy(vmin).to(value.device)
            denom = torch.from_numpy(denom).to(value.device)
        return value * denom + vmin

    elif mode == "mean_std":
        mean = np.array(key_stats["mean"], dtype=np.float32)
        std = np.maximum(np.array(key_stats["std"], dtype=np.float32), 1e-8)
        if is_tensor:
            mean = torch.from_numpy(mean).to(value.device)
            std = torch.from_numpy(std).to(value.device)
        return value * std + mean

    else:
        raise ValueError(f"Unknown normalization mode: {mode}")
