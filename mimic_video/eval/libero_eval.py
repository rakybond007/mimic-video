"""
Single-process LIBERO evaluation with frame buffer support.

Supports:
- Multi-frame observation history (num_frames > 1)
- Action chunking with replan_step
- Proper frame buffer management during action execution

Follows GR00T's evaluation protocol from
Isaac-GR00T-AlinVLA/examples/Libero/eval/run_libero_eval.py.
"""

import os
from collections import deque
from dataclasses import dataclass, field
from typing import Any

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


@dataclass
class ObservationBuffer:
    """
    Sliding window buffer for multi-frame observations.

    Stores the last `num_frames` observations (video + state) for temporal context.
    When model expects T frames, this buffer ensures we always have T frames ready.

    Frame buffer logic:
    - Initial fill: Repeat first observation until buffer is full
    - During rollout: Push new observation, oldest one drops off
    - Output shape: video (T, H, W, C), state (T, dim)
    """
    num_frames: int = 8

    # Internal storage
    front_views: deque = field(default_factory=deque)
    wrist_views: deque = field(default_factory=deque)
    states: deque = field(default_factory=deque)

    def __post_init__(self):
        self.front_views = deque(maxlen=self.num_frames)
        self.wrist_views = deque(maxlen=self.num_frames)
        self.states = deque(maxlen=self.num_frames)

    def reset(self):
        """Clear the buffer."""
        self.front_views.clear()
        self.wrist_views.clear()
        self.states.clear()

    def push(self, front_view: np.ndarray, wrist_view: np.ndarray, state: dict):
        """
        Add a new observation to the buffer.

        Args:
            front_view: (H, W, 3) uint8 image
            wrist_view: (H, W, 3) uint8 image
            state: dict with eef_pos, eef_rot, gripper
        """
        self.front_views.append(front_view.copy())
        self.wrist_views.append(wrist_view.copy())
        self.states.append(state.copy())

    def is_full(self) -> bool:
        """Check if buffer has num_frames observations."""
        return len(self.front_views) >= self.num_frames

    def fill_initial(self, front_view: np.ndarray, wrist_view: np.ndarray, state: dict):
        """
        Fill the buffer with repeated initial observation.

        Called at episode start when we don't have enough history yet.
        """
        self.reset()
        for _ in range(self.num_frames):
            self.push(front_view, wrist_view, state)

    def get_video_frames(self) -> tuple[np.ndarray, np.ndarray]:
        """
        Get stacked video frames.

        Returns:
            front_views: (T, H, W, 3) uint8
            wrist_views: (T, H, W, 3) uint8
        """
        return (
            np.stack(list(self.front_views), axis=0),
            np.stack(list(self.wrist_views), axis=0),
        )

    def get_current_state(self) -> dict:
        """Get the most recent state (for model input, uses last timestep)."""
        return self.states[-1] if self.states else {}


def eval_libero(policy, cfg: dict) -> dict:
    """
    Run LIBERO evaluation with frame buffer support.

    Args:
        policy: MimicVideoPolicy instance with get_action() method
        cfg: Dict with evaluation parameters:
            - task_suite_name: str (libero_spatial, libero_object, etc.)
            - num_trials_per_task: int (default 50)
            - num_steps_wait: int (default 10)
            - num_frames: int (default 1) - number of past frames for temporal context
            - replan_step: int (default 1) - execute this many actions before replanning
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
    num_frames = cfg.get("num_frames", 1)
    replan_step = cfg.get("replan_step", 1)
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
    print(f"  num_frames: {num_frames}, replan_step: {replan_step}")
    log_file.write(f"Task suite: {task_suite_name}\n")
    log_file.write(f"num_frames: {num_frames}, replan_step: {replan_step}\n")

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

            # Reset environment and observation buffer
            env.reset()
            obs = env.set_init_state(initial_states[episode_idx])
            obs_buffer = ObservationBuffer(num_frames=num_frames)

            t = 0
            top_view_recording = []
            wrist_view_recording = []
            done = False
            action_chunk = None
            action_idx = 0

            while t < max_steps + num_steps_wait:
                try:
                    # Warmup: let objects settle
                    if t < num_steps_wait:
                        obs, reward, done, info = env.step(get_libero_dummy_action())
                        t += 1
                        continue

                    # Get preprocessed images and state
                    img, wrist_img = get_libero_image(obs)
                    state = _extract_state(obs)

                    # Record for video
                    top_view_recording.append(img)
                    wrist_view_recording.append(wrist_img)

                    # Initialize or update observation buffer
                    if not obs_buffer.is_full():
                        obs_buffer.fill_initial(img, wrist_img, state)
                    else:
                        obs_buffer.push(img, wrist_img, state)

                    # Check if we need to replan (query model for new action chunk)
                    need_replan = (action_chunk is None) or (action_idx >= replan_step)

                    if need_replan:
                        # Build observation dict with frame history
                        obs_dict = _build_observation_dict_with_buffer(
                            obs_buffer, task_description
                        )

                        # Query policy for action chunk
                        action_chunk = policy.get_action(obs_dict)
                        action_idx = 0

                    # Convert to LIBERO action format (use action at current index)
                    action = _convert_to_libero_action(action_chunk, idx=action_idx)
                    action_idx += 1

                    # Execute action
                    obs, reward, done, info = env.step(action.tolist())

                    if done:
                        task_successes += 1
                        total_successes += 1
                        break

                    t += 1

                except Exception as e:
                    print(f"Caught exception: {e}")
                    import traceback
                    traceback.print_exc()
                    log_file.write(f"Caught exception: {e}\n")
                    break

            task_episodes += 1
            total_episodes += 1

            # Save replay video
            if save_videos and top_view_recording:
                save_rollout_video(
                    top_view_recording, wrist_view_recording, total_episodes,
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


def _extract_state(obs) -> dict:
    """Extract state dict from LIBERO observation."""
    xyz = obs["robot0_eef_pos"]
    rpy = quat2axisangle(obs["robot0_eef_quat"])
    gripper = obs["robot0_gripper_qpos"]

    return {
        "eef_pos": np.array(xyz, dtype=np.float32),
        "eef_rot": np.array(rpy, dtype=np.float32),
        "gripper": np.array(gripper, dtype=np.float32),
    }


def _build_observation_dict_with_buffer(
    obs_buffer: ObservationBuffer,
    task_description: str,
) -> dict:
    """
    Build observation dict from frame buffer.

    Returns dict with:
        - video.image: (T, H, W, 3) - T frames from buffer
        - video.wrist_image: (T, H, W, 3) - T frames from buffer
        - state.*: current state (last timestep)
        - annotation.*: language instruction
    """
    front_views, wrist_views = obs_buffer.get_video_frames()
    state = obs_buffer.get_current_state()

    return {
        "video.image": front_views,                    # (T, H, W, 3)
        "video.wrist_image": wrist_views,              # (T, H, W, 3)
        "state.eef_pos_absolute": state["eef_pos"],    # (3,)
        "state.eef_rot_absolute": state["eef_rot"],    # (3,)
        "state.gripper_close": state["gripper"],       # (2,)
        "annotation.human.action.task_description": [task_description],
    }


def _convert_to_libero_action(
    action_chunk: dict[str, np.ndarray],
    idx: int = 0,
) -> np.ndarray:
    """
    Convert MimicVideoPolicy action chunk to LIBERO 7-dim format.

    Args:
        action_chunk: Dict from policy.get_action():
            - "action.eef_pos_delta": (horizon, 3)
            - "action.eef_rot_delta": (horizon, 3)
            - "action.gripper_close": (horizon, 1)
        idx: Which timestep from the action horizon to use

    Returns:
        7-dim numpy array: [dx, dy, dz, droll, dpitch, dyaw, gripper]
    """
    # Extract action components at specified index
    pos_delta = action_chunk["action.eef_pos_delta"][idx]  # (3,)
    rot_delta = action_chunk["action.eef_rot_delta"][idx]  # (3,)
    gripper = action_chunk["action.gripper_close"][idx]    # (1,)

    # Concatenate into 7-dim action
    action_array = np.concatenate([
        pos_delta.flatten(),   # dx, dy, dz
        rot_delta.flatten(),   # droll, dpitch, dyaw
        gripper.flatten(),     # gripper
    ]).astype(np.float32)

    # Normalize gripper action
    action_array = normalize_gripper_action(action_array, binarize=True)

    assert len(action_array) == 7, f"Expected 7-dim action, got {len(action_array)}"
    return action_array


# Legacy function for backward compatibility
def _build_observation_dict(obs, task_description: str) -> dict:
    """Convert LIBERO observation to MimicVideoPolicy input format (single frame)."""
    img, wrist_img = get_libero_image(obs)

    xyz = obs["robot0_eef_pos"]
    rpy = quat2axisangle(obs["robot0_eef_quat"])
    gripper = obs["robot0_gripper_qpos"]

    return {
        "video.image": np.expand_dims(img, axis=0),        # (1, H, W, 3)
        "video.wrist_image": np.expand_dims(wrist_img, axis=0),  # (1, H, W, 3)
        "state.eef_pos_absolute": np.array(xyz, dtype=np.float32),
        "state.eef_rot_absolute": np.array(rpy, dtype=np.float32),
        "state.gripper_close": np.array(gripper, dtype=np.float32),
        "annotation.human.action.task_description": [task_description],
    }
