export PYTHONPATH=$PYTHONPATH:../../lerobot/src/
export PYTHONPATH=$PYTHONPATH:../../../HIL-RL
# [optional, in case do not install xtele and xrocs]
export PYTHONPATH=$PYTHONPATH:../../rl_envs/
export PYTHONPATH=$PYTHONPATH:../../rl_envs/xtele



# task_name=close_trashbin_franka_1028
task_name=fold_rag
# task_name=hang_clothes

mkdir -p experiments/${task_name}
cd experiments/${task_name}


# python3 ../../collect_data.py robot_type@_global_=franka task@_global_=${task_name} use_human_intervention=true ego_mode=true load_classifier=false
python3 ../../collect_data.py robot_type@_global_=ur task@_global_=${task_name} use_human_intervention=true ego_mode=false load_classifier=false
# python3 ../../collect_data.py robot_type@_global_=tienkung task@_global_=${task_name} use_human_intervention=true ego_mode=false load_classifier=false
