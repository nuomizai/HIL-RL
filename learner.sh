
export PYTHONPATH=$PYTHONPATH:../../lerobot/src/
# export PYTHONPATH=$PYTHONPATH:../../../RL-Robot-Env/
export PYTHONPATH=$PYTHONPATH:../../../HIL-RL
export PYTHONPATH=$PYTHONPATH:/home/eai/Dev/sysEAI/xRocs/xRocs
export http_proxy=http://127.0.0.1:7890 && export https_proxy=http://127.0.0.1:7890


task_name=close_trashbin_franka_1028

mkdir -p experiments/${task_name}
cd experiments/${task_name}

python3 ../../learner.py robot_type@_global_=franka task@_global_=${task_name} policy_type=silri
