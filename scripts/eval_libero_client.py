#!/usr/bin/env python3
"""
Standalone LIBERO evaluation client for MimicVideo.

This script runs in the 'libero' conda environment and communicates with a
MimicVideo policy server via HTTP. It uses ONLY Python stdlib + numpy + imageio
(no extra packages needed in the libero env).

Output format matches AlinVLA's eval_taskwise_gr00t.py:
  - rollout_{task}_{episode}_{success/failure}.mp4
  - rollout_{task}_ep00_img.png  (first frame, first episode)
  - rollout_{task}_ep00_wrist.png
  - {task_idx}_results.txt

Usage:
    python scripts/eval_libero_client.py \
        --task_suite_name libero_spatial \
        --task_idx 0 \
        --port 5555 \
        --video_out_path ./eval_output/libero_spatial

Reference: Isaac-GR00T-AlinVLA/gr00t/eval/libero/eval_taskwise_gr00t.py
"""

import base64
import collections
import json
import logging
import math
import os
import pathlib
import urllib.request

import imageio
import numpy as np
import tqdm


# ---- HTTP client (stdlib only, no extra packages) ----

def _encode_value(obj):
    """Encode numpy arrays as base64 for JSON serialization."""
    if isinstance(obj, np.ndarray):
        return {
            "__ndarray__": True,
            "data": base64.b64encode(obj.tobytes()).decode("ascii"),
            "dtype": str(obj.dtype),
            "shape": list(obj.shape),
        }
    if isinstance(obj, dict):
        return {k: _encode_value(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_encode_value(v) for v in obj]
    return obj


def _decode_value(obj):
    """Decode base64-encoded numpy arrays from JSON."""
    if isinstance(obj, dict):
        if "__ndarray__" in obj:
            data = base64.b64decode(obj["data"])
            return np.frombuffer(data, dtype=obj["dtype"]).reshape(obj["shape"]).copy()
        return {k: _decode_value(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_decode_value(v) for v in obj]
    return obj


class PolicyClient:
    """HTTP client for MimicVideo policy server. Uses only Python stdlib."""

    def __init__(self, host="localhost", port=5555, timeout=60):
        self.url = f"http://{host}:{port}"
        self.timeout = timeout

    def _post(self, data):
        body = json.dumps(_encode_value(data)).encode("utf-8")
        req = urllib.request.Request(
            self.url,
            data=body,
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=self.timeout) as resp:
            return _decode_value(json.loads(resp.read()))

    def ping(self):
        try:
            result = self._post({"endpoint": "ping"})
            return result.get("status") == "ok"
        except Exception:
            return False

    def get_action(self, observations):
        result = self._post({"endpoint": "get_action", "data": observations})
        if "error" in result:
            raise RuntimeError(f"Server error: {result['error']}")
        return result


# ---- LIBERO utilities (inline, no mimic_video imports) ----

LIBERO_DUMMY_ACTION = [0.0] * 6 + [-1.0]
LIBERO_ENV_RESOLUTION = 256

TASK_SUITE_MAX_STEPS = {
    "libero_spatial": 220,
    "libero_object": 280,
    "libero_goal": 300,
    "libero_10": 520,
    "libero_90": 400,
}


def _get_libero_env(task, resolution, seed):
    """Initialize LIBERO environment."""
    from libero.libero import get_libero_path
    from libero.libero.envs import OffScreenRenderEnv

    task_description = task.language
    task_bddl_file = os.path.join(
        get_libero_path("bddl_files"), task.problem_folder, task.bddl_file
    )
    env_args = {
        "bddl_file_name": task_bddl_file,
        "camera_heights": resolution,
        "camera_widths": resolution,
    }
    env = OffScreenRenderEnv(**env_args)
    env.seed(seed)
    return env, task_description


def _quat2axisangle(quat):
    """Convert quaternion (x,y,z,w) to axis-angle."""
    if quat[3] > 1.0:
        quat[3] = 1.0
    elif quat[3] < -1.0:
        quat[3] = -1.0
    den = np.sqrt(1.0 - quat[3] * quat[3])
    if math.isclose(den, 0.0):
        return np.zeros(3)
    return (quat[:3] * 2.0 * math.acos(quat[3])) / den


# ---- Main evaluation ----

def eval_libero(args):
    """Run LIBERO evaluation following AlinVLA's protocol."""
    from libero.libero import benchmark

    np.random.seed(args.seed)
    logging.basicConfig(level=logging.INFO)

    # Initialize task suite
    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite = benchmark_dict[args.task_suite_name]()
    num_tasks = task_suite.n_tasks
    max_steps = TASK_SUITE_MAX_STEPS.get(args.task_suite_name, 400)

    logging.info(f"Task suite: {args.task_suite_name}, idx: {args.task_idx}")
    logging.info(f"Number of tasks: {num_tasks}")
    logging.info(f"Number of trials per task: {args.num_trials_per_task}")

    pathlib.Path(args.video_out_path).mkdir(parents=True, exist_ok=True)

    client = None
    total_episodes, total_successes = 0, 0

    for task_id in tqdm.tqdm(range(num_tasks), desc="Tasks"):
        # If task_idx specified, only run that task
        if args.task_idx != -1 and task_id != args.task_idx:
            continue

        task = task_suite.get_task(task_id)
        initial_states = task_suite.get_task_init_states(task_id)
        env, task_description = _get_libero_env(task, LIBERO_ENV_RESOLUTION, args.seed)
        task_segment = task_description.replace(" ", "_")

        task_episodes, task_successes = 0, 0

        for episode_idx in tqdm.tqdm(
            range(args.num_trials_per_task), desc=f"Task {task_id}", leave=False
        ):
            # Skip if video already exists (supports resume)
            fail_path = pathlib.Path(args.video_out_path) / f"rollout_{task_segment}_{episode_idx}_failure.mp4"
            succ_path = pathlib.Path(args.video_out_path) / f"rollout_{task_segment}_{episode_idx}_success.mp4"
            if fail_path.exists():
                logging.info(f"Video exists, skipping episode {episode_idx}...")
                total_episodes += 1
                task_episodes += 1
                continue
            elif succ_path.exists():
                logging.info(f"Video exists (success), skipping episode {episode_idx}...")
                total_episodes += 1
                task_episodes += 1
                total_successes += 1
                task_successes += 1
                continue

            # Connect to server lazily
            if client is None:
                client = PolicyClient(host=args.host, port=args.port)
                logging.info(f"Connecting to server at {args.host}:{args.port}...")
                if not client.ping():
                    raise ConnectionError(f"Cannot ping server at {args.host}:{args.port}")
                logging.info("Connected.")

            action_plan = collections.deque()

            # Reset environment
            env.reset()
            obs = env.set_init_state(initial_states[episode_idx])

            t = 0
            replay_images = []
            done = False

            logging.info(f"\nTask: {task_description} | Episode {episode_idx}")

            while t < max_steps + args.num_steps_wait:
                try:
                    # Warmup: let objects settle
                    if t < args.num_steps_wait:
                        obs, reward, done, info = env.step(LIBERO_DUMMY_ACTION)
                        t += 1
                        continue

                    # Get preprocessed images (rotate 180 degrees to match training)
                    img = np.ascontiguousarray(obs["agentview_image"][::-1, ::-1])
                    wrist_img = np.ascontiguousarray(
                        obs["robot0_eye_in_hand_image"][::-1, ::-1]
                    )

                    # Save first-frame images for debugging (first episode only)
                    if t == args.num_steps_wait and episode_idx == 0:
                        img_prefix = pathlib.Path(args.video_out_path) / f"rollout_{task_segment}_ep{episode_idx:02d}"
                        imageio.imwrite(f"{img_prefix}_img.png", img)
                        imageio.imwrite(f"{img_prefix}_wrist.png", wrist_img)

                    # Save for replay video
                    replay_images.append(img)

                    if not action_plan:
                        # Build observation dict for server
                        element = {
                            "video.front_view": np.array([img]),
                            "video.left_wrist_view": np.array([wrist_img]),
                            "state.eef_pos_absolute": obs["robot0_eef_pos"],
                            "state.eef_rot_absolute": _quat2axisangle(obs["robot0_eef_quat"]),
                            "state.gripper_close": obs["robot0_gripper_qpos"],
                            "annotation.human.action.task_description": [str(task_description)],
                        }

                        # Query server
                        action_chunk = client.get_action(element)

                        # Concatenate action components: (horizon, 7)
                        actions = np.concatenate([
                            action_chunk["action.eef_pos_delta"],
                            action_chunk["action.eef_rot_delta"],
                            action_chunk["action.gripper_close"].reshape(-1, 1),
                        ], axis=-1)

                        assert len(actions) >= args.replan_steps, (
                            f"Need {args.replan_steps} replan steps but got {len(actions)}"
                        )
                        action_plan.extend(actions[:args.replan_steps])

                    action = action_plan.popleft()

                    # Binarize gripper action
                    action[-1] = 1.0 if action[-1] >= 0 else -1.0

                    # Execute action
                    obs, reward, done, info = env.step(action.tolist())
                    if done:
                        task_successes += 1
                        total_successes += 1
                        break
                    t += 1

                except Exception as e:
                    logging.error(f"Exception: {e}")
                    import traceback
                    traceback.print_exc()
                    break

            task_episodes += 1
            total_episodes += 1

            # Save replay video
            suffix = "success" if done else "failure"
            if replay_images:
                video_path = (
                    pathlib.Path(args.video_out_path)
                    / f"rollout_{task_segment}_{episode_idx}_{suffix}.mp4"
                )
                imageio.mimwrite(
                    str(video_path),
                    [np.asarray(x) for x in replay_images],
                    fps=30,
                )

            logging.info(f"Success: {done}")
            logging.info(
                f"Episodes: {total_episodes}, "
                f"Successes: {total_successes} ({total_successes / total_episodes * 100:.1f}%)"
            )

        # Per-task results
        if task_episodes > 0:
            logging.info(
                f"Task {task_id} success rate: "
                f"{float(task_successes) / float(task_episodes):.3f}"
            )

        env.close()

    # Save results
    if total_episodes > 0:
        overall_rate = float(total_successes) / float(total_episodes)
        logging.info(f"Total success rate: {overall_rate:.3f}")
        logging.info(f"Total episodes: {total_episodes}")

        result_path = pathlib.Path(args.video_out_path) / f"{args.task_idx}_results.txt"
        with open(result_path, "w") as f:
            f.write(f"Total success rate: {overall_rate}\n")
            f.write(f"Total episodes: {total_episodes}\n")
        logging.info(f"Results saved to {result_path}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="LIBERO eval client for MimicVideo")

    # Server connection
    parser.add_argument("--host", type=str, default="localhost")
    parser.add_argument("--port", type=int, default=5555)

    # LIBERO settings
    parser.add_argument(
        "--task_suite_name", type=str, default="libero_spatial",
        choices=["libero_spatial", "libero_object", "libero_goal", "libero_10", "libero_90"],
    )
    parser.add_argument("--task_idx", type=int, default=-1, help="Task index (-1 = all)")
    parser.add_argument("--num_trials_per_task", type=int, default=50)
    parser.add_argument("--num_steps_wait", type=int, default=10)
    parser.add_argument("--replan_steps", type=int, default=5, help="Steps before replanning")

    # Output
    parser.add_argument("--video_out_path", type=str, default="./eval_output")

    # Misc
    parser.add_argument("--seed", type=int, default=7)

    args = parser.parse_args()
    eval_libero(args)
