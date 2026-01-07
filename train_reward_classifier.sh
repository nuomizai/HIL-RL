export PYTHONPATH=$PYTHONPATH:../../../lerobot/src/
export PYTHONPATH=$PYTHONPATH:../../../RL-Robot-Env/
export PYTHONPATH=$PYTHONPATH:../../../HIL-RL
export PYTHONPATH=$PYTHONPATH:/home/eai/Dev/sysEAI/xRocs/xRocs
export http_proxy=http://127.0.0.1:8889 && export https_proxy=http://127.0.0.1:8889


task_name=close_trashbin_franka_1028
mkdir -p experiments/${task_name}
cd experiments/${task_name}

python3 ../../train_reward_classifier.py \
    --config_path ../../train_config_reward_classifier.json \
    --dataset.root="../../experiments/${task_name}/offline_dataset/${task_name}" \
    "$@"  