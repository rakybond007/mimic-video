#!/usr/bin/env python3
"""
LIBERO evaluation CLI for MimicVideo.

Usage (single-process, direct model loading):
    python scripts/eval_libero.py \
        --checkpoint_path ./checkpoints/libero/checkpoint-50000 \
        --task_suite_name libero_spatial \
        --num_trials_per_task 50

Usage (server mode - connect to running PolicyServer):
    python scripts/eval_libero.py \
        --server_mode \
        --port 5555 \
        --task_suite_name libero_spatial

Reference: Isaac-GR00T-AlinVLA/examples/Libero/eval/run_libero_eval.py
"""

import argparse
import json
import os

from mimic_video.eval.libero_eval import eval_libero


def main():
    parser = argparse.ArgumentParser(description="LIBERO evaluation for MimicVideo")

    # Model loading
    parser.add_argument(
        "--checkpoint_path",
        type=str,
        default="",
        help="Path to model checkpoint directory (single-process mode)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device for model inference",
    )

    # Server mode
    parser.add_argument(
        "--server_mode",
        action="store_true",
        help="Connect to a running PolicyServer instead of loading model directly",
    )
    parser.add_argument("--host", type=str, default="localhost", help="Server host")
    parser.add_argument("--port", type=int, default=5555, help="Server port")

    # Evaluation parameters
    parser.add_argument(
        "--task_suite_name",
        type=str,
        default="libero_spatial",
        choices=["libero_spatial", "libero_object", "libero_goal", "libero_10", "libero_90"],
    )
    parser.add_argument("--num_trials_per_task", type=int, default=50)
    parser.add_argument("--num_steps_wait", type=int, default=10)
    parser.add_argument("--denoising_steps", type=int, default=16)
    parser.add_argument("--headless", action="store_true", default=False)
    parser.add_argument("--no_save_videos", action="store_true", default=False)
    parser.add_argument("--log_dir", type=str, default="/tmp/logs")

    args = parser.parse_args()

    # Build policy
    if args.server_mode:
        # Server/client mode
        from mimic_video.eval.server import PolicyClient

        print(f"Connecting to policy server at {args.host}:{args.port}...")
        policy = PolicyClient(host=args.host, port=args.port)
        if not policy.ping():
            raise ConnectionError(
                f"Cannot connect to policy server at {args.host}:{args.port}"
            )
        print("Connected to server.")
    else:
        # Direct model loading
        if not args.checkpoint_path:
            raise ValueError("--checkpoint_path is required in single-process mode")

        from mimic_video.policy import MimicVideoPolicy

        print(f"Loading model from {args.checkpoint_path}...")
        policy = MimicVideoPolicy.from_checkpoint(
            args.checkpoint_path,
            device=args.device,
            denoising_steps=args.denoising_steps,
        )
        print("Model loaded.")

    # Build eval config
    cfg = {
        "task_suite_name": args.task_suite_name,
        "num_trials_per_task": args.num_trials_per_task,
        "num_steps_wait": args.num_steps_wait,
        "headless": args.headless,
        "save_videos": not args.no_save_videos,
        "log_dir": args.log_dir,
    }

    # Run evaluation
    results = eval_libero(policy, cfg)

    # Save results
    results_path = os.path.join(args.log_dir, f"results_{args.task_suite_name}.json")
    os.makedirs(args.log_dir, exist_ok=True)
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {results_path}")
    print(f"Overall success rate: {results['overall_success_rate']:.3f}")


if __name__ == "__main__":
    main()
