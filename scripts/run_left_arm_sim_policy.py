#!/usr/bin/env python3
"""Run PI05 policy in TeleManipulation single-left-arm dummy simulation.

Inputs:
- image: three camera streams from a LeRobot dataset episode
- state: live left-arm joint state from ROS topic

Outputs:
- left-arm joint command topic
- lightweight numeric telemetry topics for plotting/debugging
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from std_msgs.msg import Float32MultiArray


REPO_ROOT = Path(__file__).resolve().parents[1]
OPENPI_SRC = REPO_ROOT / "third_party" / "openpi" / "src"
if str(OPENPI_SRC) not in sys.path:
    sys.path.insert(0, str(OPENPI_SRC))

from lerobot.common.datasets.lerobot_dataset import LeRobotDataset  # noqa: E402
import openpi.policies.policy_config as policy_config  # noqa: E402
import openpi.training.config as training_config  # noqa: E402


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="PI05 left-arm sim deploy bridge.")
    parser.add_argument(
        "--checkpoint-dir",
        type=Path,
        default=Path("third_party/openpi/checkpoints/pi05_aloha_wbcd_lora/evorl_pi05_lora_A5_prompt_260318"),
        help="OpenPI checkpoint directory.",
    )
    parser.add_argument(
        "--config-name",
        type=str,
        default="pi05_aloha_wbcd_lora",
        help="OpenPI training config name.",
    )
    parser.add_argument(
        "--repo-id",
        type=str,
        default="pipeline_ab/A",
        help="LeRobot dataset repo id used as image source.",
    )
    parser.add_argument("--episode-index", type=int, default=0, help="Dataset episode index for image replay.")
    parser.add_argument("--prompt", type=str, default="pick and place", help="Task prompt fed to policy.")
    parser.add_argument("--device", type=str, default="cuda", help="PyTorch device for policy infer.")
    parser.add_argument("--loop-hz", type=float, default=10.0, help="Main loop frequency.")
    parser.add_argument("--state-topic", type=str, default="/left_arm/joint_states", help="Input state topic.")
    parser.add_argument("--cmd-topic", type=str, default="/left_arm/joint_cmd", help="Output command topic.")
    return parser.parse_args()


def _to_chw_uint8(img: Any) -> np.ndarray:
    arr = np.asarray(img)
    if arr.ndim != 3:
        raise ValueError(f"Image must be 3D, got shape={arr.shape}")
    if arr.dtype != np.uint8:
        arr = arr.astype(np.float32)
        if arr.max() <= 1.01:
            arr = (arr * 255.0).clip(0, 255)
        arr = arr.astype(np.uint8)
    # HWC -> CHW
    if arr.shape[0] != 3 and arr.shape[-1] == 3:
        arr = np.transpose(arr, (2, 0, 1))
    if arr.shape[0] != 3:
        raise ValueError(f"Expected CHW image with channel=3, got shape={arr.shape}")
    return arr


def _resolve_cam_key(camera_keys: list[str], preferred: list[str], tag: str) -> str:
    for want in preferred:
        if want in camera_keys:
            return want
    for key in camera_keys:
        low = key.lower()
        if tag in low:
            return key
    raise ValueError(f"Cannot resolve {tag} camera key from {camera_keys}")


def _episode_span(dataset: LeRobotDataset, episode_index: int) -> tuple[int, int]:
    epi = getattr(dataset, "episode_data_index", None)
    if isinstance(epi, dict) and "from" in epi and "to" in epi:
        start = int(epi["from"][episode_index])
        end = int(epi["to"][episode_index])
        return start, end
    ep = dataset.meta.episodes[episode_index]
    start = int(ep["dataset_from_index"])
    end = int(ep["dataset_to_index"])
    return start, end


class LeftArmPolicyBridge(Node):
    def __init__(self, args: argparse.Namespace) -> None:
        super().__init__("left_arm_policy_bridge")
        self.args = args

        cfg = training_config.get_config(args.config_name)
        self.policy = policy_config.create_trained_policy(
            cfg,
            args.checkpoint_dir,
            default_prompt=args.prompt,
            pytorch_device=args.device,
        )

        self.dataset = LeRobotDataset(repo_id=args.repo_id, delta_timestamps=None)
        camera_keys = list(self.dataset.meta.camera_keys)
        self.cam_high_key = _resolve_cam_key(
            camera_keys, ["observation.images.head_cam", "head_cam", "cam_high"], "head"
        )
        self.cam_left_key = _resolve_cam_key(
            camera_keys, ["observation.images.left_wrist_cam", "left_wrist_cam", "cam_left_wrist"], "left"
        )
        self.cam_right_key = _resolve_cam_key(
            camera_keys, ["observation.images.right_wrist_cam", "right_wrist_cam", "cam_right_wrist"], "right"
        )

        start, end = _episode_span(self.dataset, args.episode_index)
        self.indices = list(range(start, end))
        if not self.indices:
            raise ValueError(f"Episode {args.episode_index} has no frames.")
        self.frame_ptr = 0

        self.latest_left_state: np.ndarray | None = None
        self.state_sub = self.create_subscription(JointState, args.state_topic, self._on_state, 10)
        self.cmd_pub = self.create_publisher(JointState, args.cmd_topic, 10)

        self.current_state_pub = self.create_publisher(Float32MultiArray, "/sim_policy/current_state", 10)
        self.target_action_pub = self.create_publisher(Float32MultiArray, "/sim_policy/target_action", 10)
        self.error_pub = self.create_publisher(Float32MultiArray, "/sim_policy/state_action_error", 10)

        self.timer = self.create_timer(1.0 / args.loop_hz, self._step)
        self.last_log = time.monotonic()

        self.get_logger().info(
            "Left-arm bridge started | "
            f"checkpoint={args.checkpoint_dir} | "
            f"dataset={args.repo_id} | "
            f"episode={args.episode_index} | "
            f"cams=[{self.cam_high_key},{self.cam_left_key},{self.cam_right_key}]"
        )

    def _on_state(self, msg: JointState) -> None:
        name_to_pos = dict(zip(msg.name, msg.position))
        left7 = np.array(
            [name_to_pos.get(f"joint{i}", 0.0) for i in range(1, 7)] + [name_to_pos.get("gripper", 0.0)],
            dtype=np.float32,
        )
        self.latest_left_state = left7

    def _compose_state14(self) -> np.ndarray:
        if self.latest_left_state is None:
            raise RuntimeError("No left-arm state received yet.")
        state14 = np.zeros((14,), dtype=np.float32)
        state14[:7] = self.latest_left_state
        return state14

    def _next_images(self) -> dict[str, np.ndarray]:
        idx = self.indices[self.frame_ptr]
        self.frame_ptr = (self.frame_ptr + 1) % len(self.indices)
        item = self.dataset[idx]
        return {
            "cam_high": _to_chw_uint8(item[self.cam_high_key]),
            "cam_left_wrist": _to_chw_uint8(item[self.cam_left_key]),
            "cam_right_wrist": _to_chw_uint8(item[self.cam_right_key]),
        }

    def _publish_array(self, pub, arr: np.ndarray) -> None:
        msg = Float32MultiArray()
        msg.data = np.asarray(arr, dtype=np.float32).reshape(-1).tolist()
        pub.publish(msg)

    def _step(self) -> None:
        if self.latest_left_state is None:
            return

        state14 = self._compose_state14()
        obs = {
            "images": self._next_images(),
            "state": state14,
            "prompt": self.args.prompt,
        }
        out = self.policy.infer(obs)
        actions = np.asarray(out["actions"], dtype=np.float32)
        if actions.ndim != 2 or actions.shape[1] < 14:
            raise ValueError(f"Unexpected action shape {actions.shape}, expected [T, >=14]")
        action = actions[0, :14]

        cmd = JointState()
        cmd.name = [f"joint{i}" for i in range(1, 7)] + ["gripper"]
        cmd.position = action[:7].astype(float).tolist()
        self.cmd_pub.publish(cmd)

        self._publish_array(self.current_state_pub, state14[:7])
        self._publish_array(self.target_action_pub, action[:7])
        self._publish_array(self.error_pub, action[:7] - state14[:7])

        now = time.monotonic()
        if now - self.last_log > 1.0:
            self.last_log = now
            self.get_logger().info(
                "step ok | left_state=%s | left_action=%s",
                np.array2string(state14[:7], precision=3),
                np.array2string(action[:7], precision=3),
            )


def main() -> None:
    args = _parse_args()
    if not args.checkpoint_dir.exists():
        raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint_dir}")
    rclpy.init()
    node: LeftArmPolicyBridge | None = None
    try:
        node = LeftArmPolicyBridge(args)
        rclpy.spin(node)
    finally:
        if node is not None:
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
