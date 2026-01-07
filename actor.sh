export PYTHONPATH=$PYTHONPATH:../../lerobot/src/
export PYTHONPATH=$PYTHONPATH:../../../HIL-RL
export PYTHONPATH=$PYTHONPATH:/home/eai/Dev/sysEAI/xRocs/xRocs
export http_proxy=http://127.0.0.1:8889 && export https_proxy=http://127.0.0.1:8889



task_name=close_trashbin_franka_1028
mkdir -p experiments/${task_name}
cd experiments/${task_name}


# python3 ../../actor.py robot_type@_global_=franka task@_global_=${task_name} classifier_cfg.require_train=true use_human_intervention=true ego_mode=true policy_type=silri

# [debug]
python3 ../../actor.py robot_type@_global_=franka task@_global_=${task_name} classifier_cfg.require_train=true freeze_actor=true use_human_intervention=false ego_mode=false policy_type=silri