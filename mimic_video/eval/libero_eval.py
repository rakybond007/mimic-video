"""
Single-process LIBERO evaluation.

Follows GR00T's evaluation protocol from
Isaac-GR00T-AlinVLA/examples/Libero/eval/run_libero_eval.py.
"""

import os
import time

import numpy as np
import tqdm

from mimic_video.eval.libero_utils import (
    TASK_SUITE_MAX_STEPS,
    get_libero_dummy_action,
    get_libero_env,
    get_libero_image,
    normalize_gripper_action,
    quat2axisangle,
    save_rollout_video,
)

LOG_DIR = "/tmp/logs"


def eval_libero(policy, cfg: dict) -> dict:
    """
    Run LIBERO evaluation following GR00T's protocol.

    Args:
        policy: MimicVideoPolicy instance with get_action() method
        cfg: Dict with evaluation parameters:
            - task_suite_name: str (libero_spatial, libero_object, etc.)
            - num_trials_per_task: int (default 50)
            - num_steps_wait: int (default 10)
            - headless: bool (default True)
            - save_videos: bool (default True)
            - log_dir: str (default /tmp/logs)

    Returns:
        Dict with evaluation results.
    """
    from libero.libero import benchmark

    task_suite_name = cfg.get("task_suite_name", "libero_spatial")
    num_trials_per_task = cfg.get("num_trials_per_task", 50)
    num_steps_wait = cfg.get("num_steps_wait", 10)
    headless = cfg.get("headless", True)
    save_videos = cfg.get("save_videos", True)
    log_dir = cfg.get("log_dir", LOG_DIR)

    os.makedirs(log_dir, exist_ok=True)
    log_file = open(f"{log_dir}/libero_eval_{task_suite_name}.log", "w")

    # Initialize LIBERO task suite
    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite = benchmark_dict[task_suite_name]()
    num_tasks = task_suite.n_tasks
    max_steps = TASK_SUITE_MAX_STEPS.get(task_suite_name, 400)

    print(f"Task suite: {task_suite_name} ({num_tasks} tasks)")
    log_file.write(f"Task suite: {task_suite_name}\n")

    action_keys = ["x", "y", "z", "roll", "pitch", "yaw", "gripper"]
    total_episodes, total_successes = 0, 0
    per_task_results = {}

    for task_id in tqdm.tqdm(range(num_tasks), desc="Tasks"):
        task = task_suite.get_task(task_id)
        initial_states = task_suite.get_task_init_states(task_id)
        env, task_description = get_libero_env(task, resolution=256)

        task_episodes, task_successes = 0, 0

        for episode_idx in tqdm.tqdm(range(num_trials_per_task), desc=f"Task {task_id}", leave=False):
            print(f"\nTask: {task_description}")
            log_file.write(f"\nTask: {task_description}\n")

            # Reset environment
            env.reset()
            obs = env.set_init_state(initial_states[episode_idx])

            t = 0
            top_view = []
            wrist_view = []
            done = False

            while t < max_steps + num_steps_wait:
                try:
                    # Warmup: let objects settle
                    if t < num_steps_wait:
                        obs, reward, done, info = env.step(get_libero_dummy_action())
                        t += 1
                        continue

                    # Get preprocessed images
                    img, wrist_img = get_libero_image(obs)
                    top_view.append(img)
                    wrist_view.append(wrist_img)

                    # Build observation dict for policy
                    obs_dict = _build_observation_dict(obs, task_description)

                    # Query policy
                    action_chunk = policy.get_action(obs_dict)

                    # Convert to LIBERO action format
                    action = _convert_to_libero_action(action_chunk, action_keys, idx=0)

                    # Execute action
                    obs, reward, done, info = env.step(action.tolist())

                    if done:
                        task_successes += 1
                        total_successes += 1
                        break

                    t += 1

                except Exception as e:
                    print(f"Caught exception: {e}")
                    log_file.write(f"Caught exception: {e}\n")
                    break

            task_episodes += 1
            total_episodes += 1

            # Save replay video
            if save_videos and top_view:
                save_rollout_video(
                    top_view, wrist_view, total_episodes,
                    success=done, task_description=task_description,
                    log_file=log_file,
                )

            # Log results
            success_rate = total_successes / total_episodes * 100
            print(f"Success: {done}")
            print(f"Episodes: {total_episodes}, Successes: {total_successes} ({success_rate:.1f}%)")
            log_file.write(f"Success: {done}\n")
            log_file.write(f"Episodes: {total_episodes}, Successes: {total_successes} ({success_rate:.1f}%)\n")
            log_file.flush()

        task_rate = float(task_successes) / float(task_episodes) if task_episodes > 0 else 0.0
        per_task_results[task_description] = {
            "success_rate": task_rate,
            "successes": task_successes,
            "episodes": task_episodes,
        }
        print(f"Task success rate: {task_rate:.3f}")
        log_file.write(f"Task success rate: {task_rate:.3f}\n")
        log_file.flush()

        env.close()

    overall_rate = float(total_successes) / float(total_episodes) if total_episodes > 0 else 0.0
    print(f"\nOverall success rate: {overall_rate:.3f} ({total_successes}/{total_episodes})")
    log_file.write(f"\nOverall success rate: {overall_rate:.3f}\n")
    log_file.close()

    return {
        "overall_success_rate": overall_rate,
        "total_successes": total_successes,
        "total_episodes": total_episodes,
        "per_task": per_task_results,
    }


def _build_observation_dict(obs, task_description: str) -> dict:
    """Convert LIBERO observation to MimicVideoPolicy input format."""
    img, wrist_img = get_libero_image(obs)

    xyz = obs["robot0_eef_pos"]
    rpy = quat2axisangle(obs["robot0_eef_quat"])
    gripper = obs["robot0_gripper_qpos"]

    return {
        "video.image": np.expand_dims(img, axis=0),        # (1, H, W, 3)
        "video.wrist_image": np.expand_dims(wrist_img, axis=0),  # (1, H, W, 3)
        "state.x": np.array([[xyz[0]]]),
        "state.y": np.array([[xyz[1]]]),
        "state.z": np.array([[xyz[2]]]),
        "state.roll": np.array([[rpy[0]]]),
        "state.pitch": np.array([[rpy[1]]]),
        "state.yaw": np.array([[rpy[2]]]),
        "state.gripper": np.expand_dims(gripper, axis=0),
        "annotation.human.action.task_description": [task_description],
    }


def _convert_to_libero_action(
    action_chunk: dict[str, np.ndarray],
    action_keys: list[str],
    idx: int = 0,
) -> np.ndarray:
    """
    Convert MimicVideoPolicy action chunk to LIBERO 7-dim format.

    Args:
        action_chunk: Dict from policy.get_action(), e.g. {"action.x": (horizon,), ...}
        action_keys: Key order ["x", "y", "z", "roll", "pitch", "yaw", "gripper"]
        idx: Which timestep from the action horizon to use (default: 0)

    Returns:
        7-dim numpy array: [dx, dy, dz, droll, dpitch, dyaw, gripper]
    """
    action_components = []
    for key in action_keys:
        val = action_chunk.get(f"action.{key}")
        if val is None:
            # Try alternative key formats
            for alt_key in [f"action.eef_pos_delta", f"action.eef_rot_delta", f"action.gripper_close"]:
                if alt_key in action_chunk:
                    val = action_chunk[alt_key]
                    break
        if val is None:
            val = np.zeros(1)

        component = np.atleast_1d(val[idx] if len(val) > idx else val[-1])
        action_components.append(component[0])

    action_array = np.array(action_components, dtype=np.float32)
    action_array = normalize_gripper_action(action_array, binarize=True)
    assert len(action_array) == 7, f"Expected 7-dim action, got {len(action_array)}"
    return action_array
