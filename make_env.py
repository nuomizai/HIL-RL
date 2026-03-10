import traceback
import sys
import gymnasium as gym

from typing import TypedDict

def print_green(x):
    return print("\033[92m {}\033[00m".format(x))

def make_env(config, fake_env, use_human_intervention, classifier=False, use_gripper_penalty=False, cfg=None):
    try:  
        if config.robot_config.robot_type == "sim":
            import gym_hil  # Only import when needed for sim environments
            from rl_envs.sim_wrapper import ConvertObservationWrapper
            env = gym.make(
                config.robot_config.env_id,
                render_mode=config.robot_config.render_mode,
                image_obs=True,
                use_viewer=config.robot_config.use_viewer,
                use_gamepad=config.robot_config.use_gamepad ,
                max_episode_steps=config.robot_config.max_episode_length,  # 100 seconds * 10Hz
                controller_config_path=config.robot_config.controller_config_path,
                reward_type=config.robot_config.reward_type,
                gripper_penalty=config.robot_config.gripper_penalty,
            )

            env = ConvertObservationWrapper(env)
        else:
            from rl_envs.base_env import BaseEnv
            from rl_envs.wrappers import HumanIntervention, SERLObsWrapper, AugmentedObservationWrapper
            from rl_envs.reward_wrapper import MultiCameraBinaryRewardClassifierWrapper, GripperPenaltyWrapper

            env = BaseEnv(config=config.robot_config, fake_env=fake_env)
            
            if not fake_env and use_human_intervention:
                env = HumanIntervention(env)
            
            env = AugmentedObservationWrapper(env)
            env = SERLObsWrapper(env,proprio_keys=config.robot_config.proprio_keys, use_force=config.use_force)
            if classifier:
                env = MultiCameraBinaryRewardClassifierWrapper(env, config.robot_config.classifier_cfg, cfg=cfg)
                if use_gripper_penalty:
                    env = GripperPenaltyWrapper(env, penalty=config.robot_config.gripper_penalty)
    except Exception as e:
        print_green(f"[{type(e).__name__}] {e!r}")
        traceback.print_exc()          # full stacktrace
        sys.exit(1)
    return env
