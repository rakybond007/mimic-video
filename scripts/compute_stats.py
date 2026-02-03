#!/usr/bin/env python3
"""
Compute and save dataset normalization statistics.

Usage:
    python scripts/compute_stats.py --dataset_path /path/to/libero_dataset
"""

import argparse
import json
from pathlib import Path

from mimic_video.data.normalization import compute_dataset_stats, save_stats


def main():
    parser = argparse.ArgumentParser(description="Compute dataset normalization statistics")
    parser.add_argument(
        "--dataset_path",
        type=str,
        required=True,
        help="Path to the LeRobot-format dataset directory",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output path for stats JSON (default: dataset_path/meta/stats.json)",
    )
    parser.add_argument(
        "--keys",
        type=str,
        nargs="*",
        default=None,
        help="Only compute stats for these keys (default: all numeric columns)",
    )
    args = parser.parse_args()

    dataset_path = Path(args.dataset_path)
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset path does not exist: {dataset_path}")

    print(f"Computing statistics for dataset: {dataset_path}")
    stats = compute_dataset_stats(str(dataset_path), keys=args.keys)

    output_path = args.output or str(dataset_path / "meta" / "stats.json")
    save_stats(stats, output_path)

    print(f"Saved statistics to: {output_path}")
    print(f"Keys computed: {list(stats.keys())}")
    for key, s in stats.items():
        print(f"  {key}: count={s['count']}, dims={len(s['min'])}")


if __name__ == "__main__":
    main()
