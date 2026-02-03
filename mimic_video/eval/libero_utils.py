"""
Utility functions for LIBERO evaluation.

Ported from Isaac-GR00T-AlinVLA/examples/Libero/eval/utils.py
"""

import math
import os
import time

import numpy as np

DATE = time.strftime("%Y_%m_%d")
DATE_TIME = time.strftime("%Y_%m_%d-%H_%M_%S")


def get_libero_env(task, resolution=256):
    """Initialize and return the LIBERO environment, along with the task description."""
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
    env.seed(0)  # seed affects object positions even with fixed initial state
    return env, task_description


def get_libero_dummy_action():
    """Get dummy/no-op action for simulation warmup."""
    return [0, 0, 0, 0, 0, 0, -1]


def get_libero_image(obs):
    """Extract and preprocess images from LIBERO observations.

    Images are rotated 180 degrees to match training data preprocessing.
    """
    img = obs["agentview_image"]
    img = img[::-1, ::-1]  # rotate 180 degrees

    wrist_img = obs["robot0_eye_in_hand_image"]
    wrist_img = wrist_img[::-1, ::-1]  # rotate 180 degrees

    return img, wrist_img


def quat2axisangle(quat):
    """
    Convert quaternion to axis-angle format.

    Copied from robosuite transform_utils.py.

    Args:
        quat: (x, y, z, w) quaternion

    Returns:
        (ax, ay, az) axis-angle exponential coordinates
    """
    # clip quaternion
    if quat[3] > 1.0:
        quat[3] = 1.0
    elif quat[3] < -1.0:
        quat[3] = -1.0

    den = np.sqrt(1.0 - quat[3] * quat[3])
    if math.isclose(den, 0.0):
        return np.zeros(3)

    return (quat[:3] * 2.0 * math.acos(quat[3])) / den


def normalize_gripper_action(action, binarize=True):
    """
    Convert gripper action from [0, 1] to [+1, -1].

    Normalization: y = 1 - 2 * (x - orig_low) / (orig_high - orig_low)
    """
    orig_low, orig_high = 0.0, 1.0
    action[..., -1] = 1 - 2 * (action[..., -1] - orig_low) / (orig_high - orig_low)

    if binarize:
        action[..., -1] = np.sign(action[..., -1])

    return action


def save_rollout_video(top_view, wrist_view, idx, success, task_description, log_file=None):
    """Save an MP4 replay of an episode with both camera views side-by-side."""
    import imageio

    rollout_dir = f"./rollouts/{DATE}"
    os.makedirs(rollout_dir, exist_ok=True)

    processed_desc = (
        task_description.lower().replace(" ", "_").replace("\n", "_").replace(".", "_")[:50]
    )
    mp4_path = (
        f"{rollout_dir}/{DATE_TIME}--episode={idx}--success={success}--task={processed_desc}.mp4"
    )

    video_writer = imageio.get_writer(mp4_path, fps=30)
    for img1, img2 in zip(top_view, wrist_view):
        combined = np.hstack((img1, img2))
        video_writer.append_data(combined)
    video_writer.close()

    print(f"Saved rollout MP4 at path {mp4_path}")
    if log_file is not None:
        log_file.write(f"Saved rollout MP4 at path {mp4_path}\n")

    return mp4_path


# Max steps per task suite (based on longest training demos)
TASK_SUITE_MAX_STEPS = {
    "libero_spatial": 220,
    "libero_object": 280,
    "libero_goal": 600,
    "libero_10": 1000,
    "libero_90": 400,
}
