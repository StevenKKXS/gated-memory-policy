import os
import sys
import time
from typing import Any

import hydra
import numpy as np
from omegaconf.omegaconf import OmegaConf, ListConfig

from env.utils.visualize_utils import (
    convert_data_to_video_parallel,
)
from env.utils.policy_config_utils import (
    extract_policy_rollout_timing,
    needs_full_image_history,
)

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


import yaml
import robotmq

# from env.modules.agents.manipulation_policy_agent import ManipulationPolicyAgent
from env.modules.agents.manipulation_policy_agent import ManipulationPolicyAgent
from env.modules.agents.robomimic_policy_agent import RobomimicPolicyAgent
from env.modules.tasks.base_task import BaseTask


if __name__ == "__main__":
    global config_name
    os.environ["HYDRA_FULL_ERROR"] = "1"
    task_name = sys.argv[1]
    print(f"Task name: {task_name}")
    print(f"Config directory: {os.listdir('config')}")
    if f"rollout_policy_{task_name}.yaml" in os.listdir("config"):
        config_name = f"rollout_policy_{task_name}"
        print(f"Using config: {config_name}")
    else:
        config_name = "rollout_policy"
    sys.argv[1] = f"task_name={task_name}"
    


@hydra.main(config_path="../config", config_name=config_name, version_base=None)
def main(cfg) -> None:
    # Ensure we're in the project root directory
    np.set_printoptions(precision=4, suppress=True, sign=" ")
    task_name: str = cfg.task_name
    assert task_name in [
        "pick_and_place_back",
        "pick_and_match_color",
        "pick_and_match_color_rand_delay",
        "push_cube",
        "fling_cloth",
        "robomimic_square",
        "robomimic_tool_hang",
        "robomimic_transport",
    ], f"Unknown task: {task_name}"

    # To instantiate the time string
    time_str = (
        cfg.time_str
    )  # This step will convert {now:%Y-%m-%d-%H-%M-%S} to actual time string
    cfg.time_str = time_str

    task_cfg = hydra.compose(config_name=f"task/{task_name}")
    if task_name.startswith("robomimic"):
        agent_cfg = hydra.compose(config_name=f"task/agent/robomimic_policy")
    else:
        agent_cfg = hydra.compose(config_name=f"task/agent/policy")

    OmegaConf.set_struct(
        task_cfg, False
    )  # Allows adding new fields for the first config
    OmegaConf.set_struct(
        agent_cfg, False
    )  # Allows adding new fields for the first config
    task_cfg = OmegaConf.merge(agent_cfg, task_cfg, cfg)
    OmegaConf.set_struct(
        task_cfg, False
    )  # Disable struct mode after merge to allow adding new fields
    print(f"task_cfg: {task_cfg}")
    if "env_meta" in task_cfg.task.env:
        freq = task_cfg.task.env.env_meta.env_kwargs.control_freq
    else:
        freq = task_cfg.task.env.control_freq

    # Initialize client to fetch policy config before instantiating the task
    policy_server_address = task_cfg.task.agent.policy_server_address
    temp_client = robotmq.RMQClient("temp_policy_client", policy_server_address)

    # Fetch policy config from server
    policy_config: dict[str, Any] = robotmq.deserialize(
        temp_client.request_with_data(
            "policy_config",
            robotmq.serialize(True),
        )
    )

    # Configure task based on policy config
    timing = extract_policy_rollout_timing(policy_config)
    if timing.source == "train_dataset":
        print("using train_dataset instead of model for legacy checkpoint compatibility")

    task_cfg.task.agent.obs_history_len = timing.obs_history_len
    task_cfg.task.agent.image_obs_frames_ids = ListConfig(
        timing.image_obs_frames_ids
    )  # During training data loading, the latest image is indexed as 0, previous frames are -k
    # During rollout, the latest image is indexed as -1, thus need to subtract 1
    task_cfg.task.agent.proprio_obs_frames_ids = ListConfig(
        timing.proprio_obs_frames_ids
    )

    print(f"task_cfg.task.agent.proprio_obs_frames_ids: {task_cfg.task.agent.proprio_obs_frames_ids}")

    task_cfg.task.agent.action_prediction_horizon = timing.action_prediction_horizon
    if needs_full_image_history(timing):
        task_cfg.task.render_all_images = True

    # Set use_relative_pose from policy config if applicable
    if (
        not task_name.startswith("robomimic")
        and "use_relative_pose" in policy_config.get("workspace", {}).get("train_dataset", {})
    ):
        task_cfg.task.agent.use_relative_pose = policy_config["workspace"]["train_dataset"]["use_relative_pose"]

    instantiated_cfg: dict[str, Any] = hydra.utils.instantiate(task_cfg)
    task: BaseTask = instantiated_cfg["task"]

    data_storage_dir = instantiated_cfg["task"].data_storage_dir
    if not os.path.exists(data_storage_dir):
        os.makedirs(data_storage_dir, exist_ok=True)

    assert isinstance(task.agent, RobomimicPolicyAgent) or isinstance(
        task.agent, ManipulationPolicyAgent
    )
    # policy_config already fetched before task instantiation

    with open(os.path.join(data_storage_dir, "policy_config.yaml"), "w") as f:
        yaml.dump(policy_config, f)
    print(
        f"Policy config saved to {os.path.join(data_storage_dir, 'policy_config.yaml')}"
    )

    dataset_config = task.agent.get_dataset_config()
    if dataset_config:
        with open(os.path.join(data_storage_dir, "dataset_config.yaml"), "w") as f:
            yaml.dump(dataset_config, f)
    print(
        f"Dataset config saved to {os.path.join(data_storage_dir, 'dataset_config.yaml')}"
    )
    start_time = time.time()
    episode_configs = []
    for episode_idx, seed in enumerate(
        range(cfg.start_seed, cfg.start_seed + cfg.episode_num)
    ):
        episode_config: dict[str, Any] = {
            "episode_idx": episode_idx,
            "seed": seed,
        }
        if task_name == "fling_cloth":
            # HACK: only rollout two extreme cases
            episode_config.update({"cloth_mass": 2.0 if episode_idx % 2 == 0 else 0.15})
        episode_configs.append(episode_config)
    
    task.agent.reset() # Manually reset the agent to keep the recorded data of each episode
    task.agent.skip_policy_reset = True
    task.run_episodes(episode_configs)
    export_file_path = task.agent.export_recorded_data(f"rollout_{time_str}")
    end_time = time.time()
    print(f"Time taken: {end_time - start_time:.2f} seconds")
    # Convert rollout data to video
    convert_data_to_video_parallel(
        task.data_storage_dir, freq_hz=freq, split_videos=True, video_type="success", add_black_screen_in_the_end=False
    )
    convert_data_to_video_parallel(
        task.data_storage_dir, freq_hz=freq, split_videos=True, video_type="failure", add_black_screen_in_the_end=False
    )

    print(f"Exported rollout data to {export_file_path}")

if __name__ == "__main__":
    main()
