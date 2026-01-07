#!/usr/bin/env python

# Script to split a LeRobot dataset into success and failure subsets based on reward threshold

import argparse
import logging
import os
import shutil
from pathlib import Path

import numpy as np
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.datasets.utils import DEFAULT_FEATURES
from lerobot.utils.constants import ACTION, DONE, OBS_STATE, REWARD

logging.basicConfig(level=logging.INFO)


def split_dataset_by_reward(
    repo_id: str,
    root: str | None = None,
    reward_threshold: float = 0.5,
    output_success_repo_id: str | None = None,
    output_failure_repo_id: str | None = None,
    output_root: str | None = None,
    task_name: str | None = None,
):
    """
    Split a LeRobot dataset into success and failure subsets based on reward threshold.
    
    Frames are classified based on their individual reward:
    - success: reward > reward_threshold
    - failure: reward <= reward_threshold
    
    Each frame is stored as a separate episode in the output datasets.
    
    Args:
        repo_id: The repository ID of the source dataset
        root: Root directory of the source dataset
        reward_threshold: Threshold for splitting (default 0.5)
        output_success_repo_id: Repository ID for success dataset
        output_failure_repo_id: Repository ID for failure dataset
        output_root: Root directory for output datasets
        task_name: Task name to use for all frames (if None, will try to get from dataset)
    """
    # Set default output repo IDs
    if output_success_repo_id is None:
        output_success_repo_id = f"{repo_id}_success"
    if output_failure_repo_id is None:
        output_failure_repo_id = f"{repo_id}_failure"
    
    # Load source dataset
    logging.info(f"Loading dataset from {root}...")
    input("Press Enter to continue...")
    dataset = LeRobotDataset(repo_id, root=root)
    
    logging.info(f"Dataset loaded with {dataset.num_episodes} episodes")
    logging.info(f"Features: {list(dataset.features.keys())}")

    # Get task name from external parameter or try to get from dataset
    if task_name is None:
        # Try to get task name from the first frame
        if dataset.num_frames > 0:
            first_frame = dataset.hf_dataset[0]
            if "task_index" in first_frame:
                task_idx = first_frame["task_index"]
                if isinstance(task_idx, (list, np.ndarray)):
                    task_idx = task_idx[0] if len(task_idx) > 0 else 0
                elif hasattr(task_idx, 'item'):
                    task_idx = task_idx.item()
                task_idx = int(task_idx)
                if task_idx in dataset.meta.tasks:
                    task_name = dataset.meta.tasks[task_idx]
                    logging.info(f"Using task name from dataset: {task_name}")
        # If still None, use empty string
        if task_name is None:
            task_name = ""
            logging.warning("No task name provided and could not get from dataset, using empty string")
    else:
        logging.info(f"Using provided task name: {task_name}")
    
    # Get frame indices based on their individual rewards
    success_frames = []
    failure_frames = []
    
    # Statistics per episode: {episode_index: {"success": count, "failure": count}}
    episode_stats = {}
    
    # Iterate through all frames
    for frame_idx in range(dataset.num_frames):
        # Get the frame data
        frame_data = dataset.hf_dataset[frame_idx]
        
        # Get episode index for this frame
        episode_idx = frame_data.get("episode_index")
        if episode_idx is not None:
            if isinstance(episode_idx, (list, np.ndarray)):
                episode_idx = episode_idx[0] if len(episode_idx) > 0 else 0
            elif hasattr(episode_idx, 'item'):
                episode_idx = episode_idx.item()
            episode_idx = int(episode_idx)
        else:
            # If episode_index is not available, try to infer from frame index
            # This is a fallback - ideally episode_index should be in the dataset
            episode_idx = 0
            logging.warning(f"Frame {frame_idx} has no episode_index, using 0 as fallback")
        
        # Initialize episode stats if not exists
        if episode_idx not in episode_stats:
            episode_stats[episode_idx] = {"success": 0, "failure": 0}
        
        # Get the reward of this frame
        reward = frame_data[REWARD]
        if isinstance(reward, (list, np.ndarray)):
            reward = reward[0] if len(reward) > 0 else 0.0
        elif not isinstance(reward, (int, float)):
            reward = float(reward) if reward is not None else 0.0
        
        # Classify based on this frame's reward
        if reward > reward_threshold:
            success_frames.append(frame_idx)
            episode_stats[episode_idx]["success"] += 1
            if frame_idx % 100 == 0:  # Log every 100 frames to avoid too much output
                logging.info(f"Frame {frame_idx}: reward={reward:.4f} -> SUCCESS")
        else:
            failure_frames.append(frame_idx)
            episode_stats[episode_idx]["failure"] += 1
            if frame_idx % 100 == 0:  # Log every 100 frames to avoid too much output
                logging.info(f"Frame {frame_idx}: reward={reward:.4f} -> FAILURE")
    
    # Output statistics for each episode
    logging.info(f"\n{'='*60}")
    logging.info(f"Statistics per episode (成功/失败 counts):")
    logging.info(f"{'='*60}")
    for ep_idx in sorted(episode_stats.keys()):
        stats = episode_stats[ep_idx]
        total = stats["success"] + stats["failure"]
        logging.info(f"Episode {ep_idx}: 成功={stats['success']}, 失败={stats['failure']}, 总计={total}")
    logging.info(f"{'='*60}")
    
    logging.info(f"\nSplit results (Overall):")
    logging.info(f"  Success frames: {len(success_frames)}")
    logging.info(f"  Failure frames: {len(failure_frames)}")
    
    # Create success dataset if there are success frames
    if success_frames:
        logging.info(f"\nCreating success dataset: {output_success_repo_id}")
        create_subset_dataset(
            source_dataset=dataset,
            frame_indices=success_frames,
            output_repo_id=output_success_repo_id,
            output_root=output_root,
            task_name=task_name,
        )
    else:
        logging.warning("No success frames found!")
    
    # Create failure dataset if there are failure frames
    if failure_frames:
        logging.info(f"\nCreating failure dataset: {output_failure_repo_id}")
        create_subset_dataset(
            source_dataset=dataset,
            frame_indices=failure_frames,
            output_repo_id=output_failure_repo_id,
            output_root=output_root,
            task_name=task_name,
        )
    else:
        logging.warning("No failure frames found!")
    
    logging.info("\nDataset splitting complete!")
    return success_frames, failure_frames


def create_subset_dataset(
    source_dataset: LeRobotDataset,
    frame_indices: list[int],
    output_repo_id: str,
    output_root: str | None = None,
    task_name: str = "",
):
    """
    Create a new dataset containing only the specified frames.
    Each frame is stored as a separate episode.
    
    Args:
        source_dataset: The source LeRobotDataset
        frame_indices: List of frame indices to include
        output_repo_id: Repository ID for the output dataset
        output_root: Root directory for output dataset
        task_name: Task name to use for all frames
    """
    # Create the output dataset with the same features
    features = source_dataset.features
    fps = source_dataset.fps
    output_root = os.path.join(output_root, output_repo_id)
    # Create new dataset
    output_dataset = LeRobotDataset.create(
        repo_id=output_repo_id,
        fps=fps,
        root=output_root,
        use_videos=True,
        image_writer_threads=4,
        image_writer_processes=0,
        features=features,
    )
    
    logging.info(f"Output dataset will be saved to: {output_dataset.root.absolute()}")
    
    # Copy each frame as a separate episode
    for new_ep_idx, old_frame_idx in enumerate(frame_indices):
        if new_ep_idx % 100 == 0:  # Log progress every 100 frames
            logging.info(f"Copying frame {old_frame_idx} -> episode {new_ep_idx} ({new_ep_idx}/{len(frame_indices)})")
        
        # Get the frame data using __getitem__ to get processed data including images
        frame_data = source_dataset[old_frame_idx]
        
        # Build frame dict with required fields only (exclude DEFAULT_FEATURES metadata fields)
        # DEFAULT_FEATURES are automatically handled by add_frame, so we should not include them
        frame = {}
        
        for key in features.keys():
            # Skip DEFAULT_FEATURES as they are handled automatically by add_frame
            if key in DEFAULT_FEATURES:
                continue
                
            if key in frame_data:
                value = frame_data[key]
                
                # Convert torch tensors to numpy arrays
                if hasattr(value, 'numpy'):
                    value = value.numpy()
                elif isinstance(value, list):
                    value = np.array(value)
                # Ensure next.reward and next.done have shape (1,) instead of ()
                if key in ("next.done", "next.reward", "complementary_info.discrete_penalty", "complementary_info.is_intervention"):
                    if isinstance(value, np.ndarray) and value.ndim == 0:
                        value = np.expand_dims(value, axis=0)
                    elif not isinstance(value, np.ndarray):
                        value = np.array([value])
                    # Ensure correct shape
                    if value.shape == ():
                        value = np.array([value.item()])
                
                frame[key] = value
        
        # Add the frame as a single-frame episode (task is required)
        output_dataset.add_frame(frame, task=task_name)
        
    # Save the episode (each frame is its own episode)
    output_dataset.save_episode()
    
    logging.info(f"Subset dataset saved to: {output_dataset.root.absolute()}")
    return output_dataset


def main():
    parser = argparse.ArgumentParser(
        description="Split a LeRobot dataset into success and failure subsets based on reward threshold"
    )
    parser.add_argument(
        "--repo_id",
        type=str,
        required=True,
        help="Repository ID of the source dataset",
    )
    parser.add_argument(
        "--root",
        type=str,
        default=None,
        help="Root directory of the source dataset",
    )
    parser.add_argument(
        "--reward_threshold",
        type=float,
        default=0.5,
        help="Reward threshold for splitting (default: 0.5)",
    )
    parser.add_argument(
        "--output_success_repo_id",
        type=str,
        default=None,
        help="Repository ID for success dataset (default: {repo_id}_success)",
    )
    parser.add_argument(
        "--output_failure_repo_id",
        type=str,
        default=None,
        help="Repository ID for failure dataset (default: {repo_id}_failure)",
    )
    parser.add_argument(
        "--output_root",
        type=str,
        default=None,
        help="Root directory for output datasets",
    )
    parser.add_argument(
        "--task",
        type=str,
        default=None,
        help="Task name to use for all frames (if not provided, will try to get from dataset)",
    )
    
    args = parser.parse_args()
    
    split_dataset_by_reward(
        repo_id=args.repo_id,
        root=args.root,
        reward_threshold=args.reward_threshold,
        output_success_repo_id=args.output_success_repo_id,
        output_failure_repo_id=args.output_failure_repo_id,
        output_root=args.output_root,
        task_name=args.task,
    )


if __name__ == "__main__":
    main()

