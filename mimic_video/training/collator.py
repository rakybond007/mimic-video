"""
Data collator for MimicVideo training batches.
"""

import torch


class MimicVideoDataCollator:
    """Collates dataset samples into batches for MimicVideo training."""

    def __call__(self, batch: list[dict]) -> dict:
        result = {
            "video": torch.stack([b["video"] for b in batch]),  # (B, V, T, C, H, W)
            "actions": torch.stack([b["actions"] for b in batch]),  # (B, horizon, max_action_dim)
            "action_mask": torch.stack([b["action_mask"] for b in batch]),  # (B, max_action_dim)
            "joint_state": torch.stack([b["joint_state"] for b in batch]),  # (B, T, max_state_dim)
            "prompts": [b["prompt"] for b in batch],  # list[str]
        }

        # Add future_video if present in batch (Algorithm 2)
        if "future_video" in batch[0] and batch[0]["future_video"] is not None:
            result["future_video"] = torch.stack([b["future_video"] for b in batch])  # (B, V, T_future, C, H, W)

        return result
