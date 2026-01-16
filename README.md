



<div align="center">

# Real-world Reinforcement Learning from <br> Suboptimal Interventions 

SiLRI: A state-wise Lagrangian RL algorithm for real-world robotic manipulation that enables efficient online learning from suboptimal interventions.

> Yinuo Zhao<sup>1,2</sup>, Huiqian Jin<sup>1,3</sup>, Lechun Jiang<sup>1,3</sup>, Xinyi Zhang<sup>1,4</sup>, Kun Wu<sup>1</sup>, Pei Ren<sup>1</sup>, Zhiyuan Xu<sup>1,&dagger;</sup>, Zhengping Che<sup>1,&dagger;</sup>, Lei Sun<sup>3</sup>, Dapeng Wu<sup>2</sup>, Chi Harold Liu<sup>4</sup>, Jian Tang<sup>1,&#9993;</sup>

<sup>1</sup>Beijing Innovation Center of Humanoid Robotics,
<sup>2</sup>City University of Hong Kong  
<sup>3</sup>Nankai University,
<sup>4</sup>Beijing Institute of Technology


<sup>&dagger;</sup>Project leader,
<sup>&#9993;</sup>Corresponding author,

[![arXiv](https://img.shields.io/badge/arXiv-2512.24288-b31b1b?logo=arxiv&logoColor=white)](https://arxiv.org/abs/2512.24288)
 [![Project Page](https://img.shields.io/badge/Project-Page-green.svg?logo=github&logoColor=white)](https://silri-rl.github.io/)


[\[📖 Documents\]](#-documents) [\[🚀 Installation\]](#-installation) [\[📖 Training Recipe\]](#-training-recipe)  [\[🙋 FAQs\]](#-faqs)



</div>

## TODO List

* [ ] **Real World:** Release the `xrocs` and `xtele` packages for UR robots.
* [ ] **Real World:** Release the Docker image and the `xrocs`/`xtele` packages for Franka.
* [ ] **Simulator:** Release simulator examples for users without a teleoperation system.



## Repository Structure

**HIL-RL** (key components)

* `actor.py`: Actor script that queries actions from `lerobot` and executes them on the robot via `rl_envs`.
* `learner.py`: Learner script that receives transitions from the actor and sends updated model parameters back.
* `train_config_silri_franka.json`: LeRobot model configuration file for training runs on the Franka robot.
* `rl_envs/`: Robot environments and wrappers (Franka and UR supported).
* `lerobot/`: Open-source RL baseline library. In addition to `HIL-SERL`, we added `SilRI` and `HG-Dagger`.

**HIL-RL** (other components)

* `collect_data.py`: Collects 20 offline demonstrations to train a task-specific reward classifier and to initialize the intervention buffer.
* `split_data.py`: Splits offline demonstrations into success and failure samples based on rewards, ensuring balanced training and fine-tuning for the reward classifier.
* `train_reward_classifier.py`: Trains the task-specific reward classifier.
* `cfg/`: Configuration files for robot and task setup.


## 📖 Documents

This repository is built upon a fork of [Lerobot](https://github.com/huggingface/lerobot) and [HIL-SERL](https://github.com/rail-berkeley/hil-serl). Unlike the original `hil-serl` and `ConRFT` JAX implementation, we reimplement all algorithms in PyTorch for improved usability and better compatibility with the robotics community.

For common questions, please refer to `docs/` for details.

## 🚀 Installation


Download our source code:
```bash
git clone --recurse-submodules https://github.com/nuomizai/HIL-RL.git
cd HIL-RL
```
Create a virtual environment, then install the dependencies:
```bash
conda create -n silri python=3.10
conda activate silri

cd lerobot
pip install -e .
cd ..

pip install torch==2.1.1 torchvision==0.16.1 torchaudio==2.1.1 --index-url https://download.pytorch.org/whl/cu121

pip install -r requirements.txt
```
Some users may need to run the following command:
```
pip uninstall torchcodec

```
## 📖 Training Recipe

The overall training pipeline follows [HIL-SERL](https://github.com/rail-berkeley/hil-serl). Specifically, real-world RL training consists of three stages:


### 📑 Stage 1: Offline Data Collection

First, collect 20 demonstration trajectories by running:

```bash
bash collect_data.sh
```

In `collect_data.sh`, set `robot_type` (configured in `cfg/robot_type`) and `task_name` (configured in `cfg/task`). You may also want to override the following parameters for your platform and task:

```yaml
# Robot (e.g., cfg/robot_type/franka.yaml)
image_crop: Crop the third-view image to focus on the region of interest.
image_keys: Camera names matching your setup.
```

```yaml
# Task (e.g., cfg/task/close_trashbin_franka_1028.yaml)
abs_pose_limit_high: Upper bound of the absolute pose limits (safety).
abs_pose_limit_low: Lower bound of the absolute pose limits (safety).
reset_joint: Joint values used for reset.
fix_gripper: Whether to keep the gripper fixed.
close_gripper: If fix_gripper is true, keep the gripper always open or always closed.
max_episode_length: Maximum steps per episode.
```


Next, split the dataset into success and failure subsets by running `bash split_data.sh` to address class imbalance during classifier training (negative samples ≫ positive samples).


### 📑 Stage 2: Reward Classifier Training

Train a task-specific reward classifier on success/failure subsets by running `bash train_reward_classifier.sh`. In the script, set `task_name` and `dataset.root`.

### 📑 Stage 3: Online RL Training

In this stage, we train a robust robot manipulation policy with RL/IL algorithms. Specifically, there are serveral key parameters you need to specify in `actor.sh`, `learner.sh`, and their corresponding configuration files:

```yaml
# actor.sh:
task_name: the task name same as that under cfg/task/, e.g., close_transhbin_franka_1028
robot_type@_global_: the robot platform same as that under cfg/robot_type, e.g., franka
classifier_cfg.require_train: continue to train the classifier during RL training, we set True for all tasks
use_human_intervention: enable human intervention during RL training, always True, set False only when debugging.
ego_mode: intervene and reset scene by one person, set False if you have others assist to reset.
policy_type: silri (ours), hgdagger(HG-Dagger), sac(HIL-SERL)
```

```yaml
# learner.sh
same as that in actor
```

```yaml
# train_config_silri_franka.json, most parameters can be override by that specified in actor.sh/learner.sh, other key parameteres are:
policy.actor_learner_config.learner_host: the ip of learner server

```

After configureing all these above parameters, first run the learner on learner server by running `bash learner.sh`, then, run the actor process on actor server by running `bash actor.sh`.





## License

This project is released under the [Apache License](LICENSE). Parts of this project contain code and models from other sources, which are subject to their respective licenses.

## Citation

If you find this project useful in your research, please consider cite:

```BibTeX
@article{zhao2025real,
  title={Real-world Reinforcement Learning from Suboptimal Interventions},
  author={Zhao, Yinuo and Jin, Huiqian and Jiang, Lechun and Zhang, Xinyi and Wu, Kun and Ren, Pei and Xu, Zhiyuan and Che, Zhengping and Sun, Lei and Wu, Dapeng and others},
  journal={arXiv preprint arXiv:2512.24288},
  year={2025}
}
```

## 🙋 FAQs
If you run into any issues, please open a GitHub issue or contact `linda.chao.007@gmail.com`. We welcome feedback and contributions!

## Acknowledgement
HIL-RL is built with reference to the code of the following projects: [Lerobot](https://github.com/huggingface/lerobot), [HIL-SERL](https://github.com/rail-berkeley/hil-serl). Thanks for their awesome work!


