# !/usr/bin/env python

# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import logging
import sys
import traceback
import time
from dataclasses import dataclass
from typing import Any

import gymnasium as gym
import numpy as np
import torch

from lerobot.cameras import opencv  # noqa: F401
from lerobot.configs import parser
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.envs.configs import HILSerlRobotEnvConfig
from lerobot.model.kinematics import RobotKinematics
from lerobot.processor import (
    EnvTransition,
    TransitionKey,
)
from lerobot.robots import (  # noqa: F401
    RobotConfig,
    make_robot_from_config,
    so100_follower,
)
from pynput import keyboard
from lerobot.utils.constants import ACTION, DONE, OBS_IMAGES, OBS_STATE, REWARD, TRUNCATED
import time
from lerobot.utils.utils import log_say
import hydra
import draccus
from lerobot.configs.train import TrainRLServerPipelineConfig
from lerobot.policies.silri.configuration_silri import SiLRIConfig  # noqa: F401
from lerobot.configs.types import PolicyFeature, FeatureType

from lerobot.robots import so100_follower  # noqa: F401
from lerobot.scripts.rl.gym_manipulator import make_robot_env
from lerobot.teleoperators import gamepad, so101_leader  # noqa: F401
from lerobot.transport import services_pb2, services_pb2_grpc
from lerobot.transport.utils import (
    bytes_to_state_dict,
    grpc_channel_options,
    python_object_to_bytes,
    receive_bytes_in_chunks,
    send_bytes_in_chunks,
    transitions_to_bytes,
)
from lerobot.utils.process import ProcessSignalHandler
from lerobot.utils.queue import get_last_item_from_queue
from lerobot.utils.random_utils import set_seed
from lerobot.utils.robot_utils import busy_wait
from lerobot.utils.transition import (
    Transition,
    move_state_dict_to_device,
    move_transition_to_device,
)
from lerobot.utils.utils import (
    TimerManager,
    get_safe_torch_device,
    init_logging,
)
from make_env import make_env
from rl_envs.shared_state import shared_state
logging.basicConfig(level=logging.INFO)

def create_transition(
    observation=None, action=None, reward=None, done=None, truncated=None, info=None, complementary_data=None
):
    """Helper to create an EnvTransition dictionary."""
    return {
        TransitionKey.OBSERVATION: observation,
        TransitionKey.ACTION: action,
        TransitionKey.REWARD: reward,
        TransitionKey.DONE: done,
        TransitionKey.TRUNCATED: truncated,
        TransitionKey.INFO: info,
        TransitionKey.COMPLEMENTARY_DATA: complementary_data,
    }


@dataclass
class DatasetConfig:
    """Configuration for dataset creation and management."""

    repo_id: str
    task: str
    root: str | None = None
    num_episodes_to_record: int = 5
    replay_episode: int | None = None
    push_to_hub: bool = False


@dataclass
class GymManipulatorConfig:
    """Main configuration for gym manipulator environment."""

    env: HILSerlRobotEnvConfig
    dataset: DatasetConfig
    mode: str | None = None  # Either "record", "replay", None
    device: str = "cpu"


def on_press(key):
    try:
        if str(key) == 'Key.scroll_lock':
            print("----------------set human intervention key to {}!----------------".format(shared_state.human_intervention_key))
            shared_state.human_intervention_key = not shared_state.human_intervention_key
            time.sleep(0.5)
        if str(key) == 'Key.space' or str(key) == 'Key.pause':
            print("----------------set terminate to true!----------------")
            shared_state.terminate = True
            time.sleep(0.5)
    except AttributeError:
        pass
try:
    listener = keyboard.Listener(
        on_press=on_press)
    listener.start()
except Exception as e:
    print("error in keyboard listener:", e)
    exit(0)




def sanitize_info_for_transition(info: dict) -> dict:
    """Sanitize info to only include types supported by Transition.complementary_info.

    Allowed types per downstream consumer: torch.Tensor, float, int, bool.
    - np.ndarray -> torch.from_numpy(...)
    - list/tuple of numbers/bools -> torch.tensor([...])
    - np.bool_/np.integer/np.floating -> Python scalar via .item()
    - torch.Tensor -> kept as is
    Unsupported types are skipped with a warning to avoid runtime errors.
    """
    safe_info = {}
    for key, value in info.items():
        try:
            if isinstance(value, torch.Tensor):
                safe_info[key] = value
            elif isinstance(value, np.ndarray):
                # Convert arrays directly to tensor; device will be handled later
                safe_info[key] = torch.from_numpy(value)
            elif isinstance(value, (list, tuple)):
                # If it's a sequence of numbers/bools, convert to tensor
                if all(isinstance(v, (int, float, bool, np.integer, np.floating, np.bool_)) for v in value):
                    safe_info[key] = torch.tensor([v.item() if isinstance(v, (np.integer, np.floating, np.bool_)) else v for v in value])
                else:
                    logging.warning(f"Dropping complementary_info[{key}] due to unsupported list/tuple contents type: {type(value)}")
            elif isinstance(value, (np.bool_, np.integer, np.floating)):
                safe_info[key] = value.item()
            elif isinstance(value, (int, float, bool)):
                safe_info[key] = value
            else:
                logging.warning(f"Dropping complementary_info[{key}] of unsupported type: {type(value)}")
        except Exception as e:
            logging.warning(f"Failed to sanitize complementary_info[{key}] ({type(value)}): {e}")
    return safe_info


def make_policy_obs(obs: dict, device: torch.device, robot_type: str) -> dict:
    # 先将numpy数组转换为Tensor，再调整维度顺序
    policy_obs = {}
    for keys in obs.keys():
        if "state" not in keys:
            img = torch.from_numpy(obs[keys]).permute(2, 0, 1).float().unsqueeze(0).to(device) / 255.
            new_key = "observation.images." + keys
            policy_obs[new_key] = img
        else:
            state = torch.from_numpy(obs[keys]).float().unsqueeze(0).to(device)
            new_key = "observation.state"
            policy_obs[new_key] = state
    return policy_obs

def step_env_and_process_transition(
    env,
    action: torch.Tensor,
    env_cfg: any,
) -> EnvTransition:
    """
    Execute one step with processor pipeline.

    Args:
        env: The robot environment
        transition: Current transition state
        action: Action to execute
        env_processor: Environment processor
        action_processor: Action processor

    Returns:
        Processed transition with updated state.
    """
    device = torch.device("cpu")


    obs, reward, terminated, truncated, info = env.step(action)
    obs = make_policy_obs(obs, device, env_cfg.robot_config.robot_type)


    new_transition = create_transition(
        observation=obs,
        action=action,
        reward=reward,
        done=terminated,
        truncated=truncated,
        complementary_data=info,
    )
    return new_transition



def control_loop(
    env: gym.Env,
    cfg: any,
    env_cfg: any,
) -> None:
    """Main control loop for robot environment interaction.
    if cfg.mode == "record": then a dataset will be created and recorded

    Args:
     env: The robot environment
     cfg: gym_manipulator configuration
    """
    device = torch.device("cpu")

    # Reset environment and processors
    obs, info = env.reset()
    print('after reset')


    # Process initial observation
    complementary_data = {
        "discrete_penalty": 0.0,
    }
    obs = make_policy_obs(obs, device, env_cfg.robot_config.robot_type)
    transition = create_transition(observation=obs, info=info, complementary_data=complementary_data)
    use_gripper = not env_cfg.robot_config.fix_gripper

    dataset = None
    if cfg.mode == "record":
        action_feature = cfg.env.features['action']
        features = {
            ACTION: {"dtype": "float32", "shape": (action_feature.shape[0]+1,), "names": None},
            REWARD: {"dtype": "float32", "shape": (1,), "names": None},
            DONE: {"dtype": "bool", "shape": (1,), "names": None},
        }
        # if use_gripper:
        features["complementary_info.discrete_penalty"] = {
            "dtype": "float32",
            "shape": (1,),
            "names": ["discrete_penalty"],
        }

        features["complementary_info.is_intervention"] = {
            "dtype": "bool",
            "shape": (1,),
            "names": ["is_intervention"],
        }

        for key, value in transition[TransitionKey.OBSERVATION].items():
            if key == OBS_STATE:
                features[key] = {
                    "dtype": "float32",
                    "shape": value.squeeze(0).shape,
                    "names": None,
                }
            if "image" in key:
                features[key] = {
                    "dtype": "video",
                    "shape": value.squeeze(0).shape,
                    "names": ["channels", "height", "width"],
                }
        # Create dataset
        dataset = LeRobotDataset.create(
            cfg.dataset.repo_id,
            cfg.env.fps,
            root=cfg.dataset.root,
            use_videos=True,
            image_writer_threads=4,
            image_writer_processes=0,
            features=features,
        )
        logging.info(f"Dataset will be saved to: {dataset.root.absolute()}")
        print('3')

    episode_idx = 0
    episode_step = 0
    episode_start_time = time.perf_counter()
    # Handle both PolicyFeature objects and dictionaries
    action_feature = cfg.env.features['action']
    if isinstance(action_feature, PolicyFeature):
        continuous_action_dim = action_feature.shape[0]
    else:
        continuous_action_dim = action_feature['shape'][0]
    
    terminate_count = 0
    episode_length_list = []
    success_frame_count = 0
    while episode_idx < cfg.dataset.num_episodes_to_record:
        # print('episode_idx:', episode_idx)
        step_start_time = time.perf_counter()

        neutral_action = torch.tensor([0.0] * (continuous_action_dim + 1), dtype=torch.float32)

        # Use the new step function
        transition = step_env_and_process_transition(
            env=env,
            action=neutral_action,
            env_cfg=env_cfg,
        )
        terminated = transition.get(TransitionKey.DONE, False)
        truncated = transition.get(TransitionKey.TRUNCATED, False)

        if cfg.mode == "record":
            # observations = transition[TransitionKey.OBSERVATION]
            observations = {
                k: v.squeeze(0).cpu()
                for k, v in transition[TransitionKey.OBSERVATION].items()
                if isinstance(v, torch.Tensor)
            }
            # Use teleop_action if available, otherwise use the action from the transition
            reward = transition[TransitionKey.REWARD]
            action_to_record = transition[TransitionKey.ACTION]
            if "is_intervention" in transition[TransitionKey.COMPLEMENTARY_DATA] and transition[TransitionKey.COMPLEMENTARY_DATA]["is_intervention"]:
                if env_cfg.robot_config.robot_type == "sim":
                    action_to_record = transition[TransitionKey.COMPLEMENTARY_DATA]["teleop_action"]
                else:
                    action_to_record = transition[TransitionKey.COMPLEMENTARY_DATA]["intervene_action"]
            else:
                print('No intervention!!!!!!!!!!!!!!!!!!!')

            frame = {
                **observations,
                ACTION: action_to_record.cpu() if isinstance(action_to_record, torch.Tensor) else action_to_record,
                REWARD: np.array([transition[TransitionKey.REWARD]], dtype=np.float32),
                DONE: np.array([terminated], dtype=bool),
                # TRUNCATED: np.array([truncated], dtype=bool),
            }
            complementary = transition.get(TransitionKey.COMPLEMENTARY_DATA, {})
            frame["complementary_info.discrete_penalty"] = np.array([complementary["discrete_penalty"]], dtype=np.float32)
            frame["complementary_info.is_intervention"] = np.array([complementary["is_intervention"]], dtype=bool)
            if dataset is not None and complementary["is_intervention"]:
                # frame["task"] = cfg.dataset.task
                if frame["next.reward"] > 0:
                    print("add one success frame into dataset")
                    success_frame_count += 1
                dataset.add_frame(frame, task=env_cfg.task_name)

        episode_step += 1
        terminated = terminated or shared_state.terminate or truncated
        if terminated :
            terminate_count += 1 
        print('terminated:', terminated, '; terminate_count:', terminate_count)
        # Handle episode termination
        if terminated and terminate_count >= 10:
            episode_time = time.perf_counter() - episode_start_time
            logging.info(
                f"Episode ended after {episode_step} steps in {episode_time:.1f}s with reward {transition[TransitionKey.REWARD]}"
            )
            episode_length_list.append(episode_step)
            episode_step = 0
            episode_idx += 1

            if dataset is not None:
                logging.info(f"Saving episode {episode_idx} with {success_frame_count} success frames")
                dataset.save_episode()

            # Reset for new episode
            obs, info = env.reset()
            obs = make_policy_obs(obs, device, env_cfg.robot_config.robot_type)
            transition = create_transition(observation=obs, info=info)
            terminate_count = 0
            shared_state.terminate = False
            success_frame_count = 0



    episode_length_list = np.array(episode_length_list)
    print("episode_length_mean:", np.mean(episode_length_list)) 
    if dataset is not None:
        logging.info(f"Dataset saved to: {dataset.root.absolute()}")
        input("Press Enter to continue...")
        if cfg.dataset.push_to_hub:
            logging.info("Pushing dataset to hub")
            dataset.push_to_hub()
    


def replay_trajectory(
    env, cfg
) -> None:
    """Replay recorded trajectory on robot environment."""
    assert cfg.dataset.replay_episode is not None, "Replay episode must be provided for replay"

    dataset = LeRobotDataset(
        cfg.dataset.repo_id,
        root=cfg.dataset.root,
        episodes=[cfg.dataset.replay_episode],
        download_videos=False,
    )
    episode_frames = dataset.hf_dataset.filter(lambda x: x["episode_index"] == cfg.dataset.replay_episode)
    actions = episode_frames.select_columns(ACTION)

    _, info = env.reset()

    for action_data in actions:
        start_time = time.perf_counter()
        transition = create_transition(
            observation=env.get_raw_joint_positions() if hasattr(env, "get_raw_joint_positions") else {},
            action=action_data[ACTION],
        )
        # transition = action_processor(transition)
        action = transition[TransitionKey.ACTION]
        env.step(transition[TransitionKey.ACTION])



@hydra.main(config_path="./cfg", config_name="config", version_base=None) 
def main(env_cfg):
    if "franka" in env_cfg.robot_config.robot_type:
        lerobot_config_path = "../../train_config_collect_data.json"
    else:
        raise ValueError(f"Invalid robot type: {env_cfg.robot_type}")

    with draccus.config_type("json"):
        cfg = draccus.parse(GymManipulatorConfig, lerobot_config_path, args=[f"--dataset.task={env_cfg.task_name}", f"--dataset.root={env_cfg.task_name}"])
    

    """Main entry point for gym manipulator script."""
    try:
        env = make_env(env_cfg, fake_env=False, use_human_intervention=env_cfg.use_human_intervention, classifier=True, use_gripper_penalty=True)
    except Exception as e:
        traceback.print_exc()          # full stacktrace
        sys.exit(1)
    print('success make env')
    if cfg.mode == "replay":
        replay_trajectory(env, cfg)
        exit(0)

    control_loop(env, cfg, env_cfg)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"In collect_data.py: [{type(e).__name__}] {e!r}")
        traceback.print_exc()          # full stacktrace
        sys.exit(1)